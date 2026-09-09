"""
app/reasoning.py — Hand-Written 3-Stage Decision Layer

Strictly implements Section 6 of the project brief:
- Stage 1: Intent Router (`route_intent`) — Raw HTTP call (httpx) to Anthropic or OpenAI.
           Extracts strict JSON: {needs_detection, query_type, target_class, negated}.
- Stage 2: Structured Reasoning (`reason_over_detections`) — Pure Python deterministic logic.
           Handles count, presence_check, most_common, and computes aggregate confidence.
- Stage 3: Confidence Guardrail (`apply_guardrail`) — Flags low confidence or insufficient evidence.

NO LangChain, NO LangGraph, NO CrewAI, NO agent frameworks.
"""

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("rtdetr-ppe.reasoning")

# Canonical class mappings and aliases
CLASS_SYNONYMS = {
    "hardhat": "Head_protection",
    "helmet": "Head_protection",
    "head": "Head_protection",
    "head_protection": "Head_protection",
    "vest": "Safety_vest",
    "safety_vest": "Safety_vest",
    "safety-vest": "Safety_vest",
    "hi-vis": "Safety_vest",
    "hivis": "Safety_vest",
    "goggles": "Eye_protection",
    "glasses": "Eye_protection",
    "eye": "Eye_protection",
    "eye_protection": "Eye_protection",
    "boots": "Foot_protection",
    "shoes": "Foot_protection",
    "foot": "Foot_protection",
    "foot_protection": "Foot_protection",
    "gloves": "Hand_protection",
    "hand": "Hand_protection",
    "hand_protection": "Hand_protection",
    "mask": "Respiratory_protection",
    "respirator": "Respiratory_protection",
    "respiratory_protection": "Respiratory_protection",
    "person": "Person",
    "worker": "Person",
    "people": "Person",
    "workers": "Person",
}

NEGATIVE_PAIRS = {
    "Head_protection": "No_head_protection",
    "Safety_vest": "No_safety_vest",
    "Eye_protection": "No_eye_protection",
    "Foot_protection": "No_foot_protection",
    "Hand_protection": "No_hand_protection",
    "Respiratory_protection": "No_respiratory_protection",
}


def normalize_class_name(raw_name: Optional[str]) -> Optional[str]:
    """Map informal/user class names to official dataset class labels."""
    if not raw_name:
        return None
    cleaned = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_SYNONYMS.get(cleaned, raw_name)


# ── Stage 1: Intent Router (Raw HTTP Call to LLM) ────────────────────────────

SYSTEM_PROMPT = """You are a precise query router for a construction-site PPE compliance system.
The system detects 13 classes:
- Person
- Head_protection, No_head_protection (hardhats/helmets)
- Safety_vest, No_safety_vest (safety vests)
- Eye_protection, No_eye_protection (goggles/glasses)
- Foot_protection, No_foot_protection (safety boots/shoes)
- Hand_protection, No_hand_protection (gloves)
- Respiratory_protection, No_respiratory_protection (masks/respirators)

Analyze the user's natural language question and output STRICT JSON ONLY with NO preamble or markdown:
{
  "needs_detection": boolean,
  "query_type": "count" | "presence_check" | "most_common" | "general_no_detection_needed",
  "target_class": "Person" | "Head_protection" | "Safety_vest" | "Eye_protection" | "Foot_protection" | "Hand_protection" | "Respiratory_protection" | null,
  "negated": boolean
}

Rules:
- "Is anyone not wearing a helmet?" -> query_type="presence_check", target_class="Head_protection", negated=true
- "Is everyone wearing a vest?" -> query_type="presence_check", target_class="Safety_vest", negated=true (checking non-compliance)
- "How many workers are wearing hardhats?" -> query_type="count", target_class="Head_protection", negated=false
- "How many people are on site?" -> query_type="count", target_class="Person", negated=false
- "What PPE is most common?" -> query_type="most_common", target_class=null, negated=false
- "What is the capital of France?" -> query_type="general_no_detection_needed", needs_detection=false, target_class=null, negated=false
"""


def _rule_based_fallback_intent(question: str) -> Dict[str, Any]:
    """Deterministic fallback router when no external LLM API key is set."""
    q = question.lower()

    # General / off-topic check
    ppe_keywords = [
        "helmet", "hardhat", "vest", "goggle", "glass", "boot", "shoe", "glove",
        "mask", "worker", "person", "people", "wear", "ppe", "safety", "site"
    ]
    if not any(kw in q for kw in ppe_keywords):
        return {
            "needs_detection": False,
            "query_type": "general_no_detection_needed",
            "target_class": None,
            "negated": False,
        }

    # Negation check
    negated = any(neg in q for neg in ["not", "no ", "without", "missing", "unprotected", "bare"])

    # Target class check
    target_class = "Person"
    if any(w in q for w in ["helmet", "hardhat", "head"]):
        target_class = "Head_protection"
    elif any(w in q for w in ["vest", "hi-vis", "jacket"]):
        target_class = "Safety_vest"
    elif any(w in q for w in ["goggle", "glass", "eye"]):
        target_class = "Eye_protection"
    elif any(w in q for w in ["boot", "shoe", "foot"]):
        target_class = "Foot_protection"
    elif any(w in q for w in ["glove", "hand"]):
        target_class = "Hand_protection"
    elif any(w in q for w in ["mask", "respirator"]):
        target_class = "Respiratory_protection"

    # Query type check
    if any(w in q for w in ["how many", "count", "number of"]):
        query_type = "count"
    elif any(w in q for w in ["most common", "frequent", "mostly"]):
        query_type = "most_common"
        target_class = None
    else:
        query_type = "presence_check"

    return {
        "needs_detection": True,
        "query_type": query_type,
        "target_class": target_class,
        "negated": negated,
    }


def route_intent(question: str, timeout: float = 8.0) -> Dict[str, Any]:
    """
    Stage 1: Intent Router.
    Makes a single, direct raw HTTP call (httpx) to Anthropic or OpenAI API.
    Falls back gracefully to deterministic rule-based router if API keys are absent.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Try Anthropic API via direct HTTP POST
    if anthropic_key:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 256,
                        "temperature": 0.0,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": question}],
                    },
                )
                if resp.status_code == 200:
                    content = resp.json()["content"][0]["text"].strip()
                    # Clean potential markdown formatting
                    content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE)
                    data = json.loads(content)
                    data["target_class"] = normalize_class_name(data.get("target_class"))
                    return data
        except Exception as e:
            logger.warning(f"Anthropic direct HTTP router failed: {e}. Falling back.")

    # Try OpenAI API via direct HTTP POST
    if openai_key:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ],
                    },
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    data = json.loads(content)
                    data["target_class"] = normalize_class_name(data.get("target_class"))
                    return data
        except Exception as e:
            logger.warning(f"OpenAI direct HTTP router failed: {e}. Falling back.")

    # Deterministic local fallback
    return _rule_based_fallback_intent(question)


# ── Stage 2: Structured Reasoning (Deterministic Python Logic) ───────────────

def reason_over_detections(detections: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: Deterministic Pure Python Reasoning.
    Executes business logic strictly over detection outputs based on intent slots.
    """
    if not intent.get("needs_detection", True):
        return {
            "answer": "This question does not appear to relate to PPE safety compliance or worker presence.",
            "used_detection": False,
            "confidence_val": 1.0,
            "supporting_detections": [],
        }

    query_type = intent.get("query_type", "presence_check")
    target_class = intent.get("target_class")
    negated = intent.get("negated", False)

    person_count = sum(1 for d in detections if d["class"] == "Person")

    # 1. Count query
    if query_type == "count":
        if target_class:
            if negated:
                neg_class = NEGATIVE_PAIRS.get(target_class, f"No_{target_class.lower()}")
                matched = [d for d in detections if d["class"].lower() == neg_class.lower()]
                readable_name = target_class.replace("_", " ").lower()
                answer = f"{len(matched)} instance(s) detected without {readable_name}."
            else:
                matched = [d for d in detections if d["class"].lower() == target_class.lower()]
                readable_name = target_class.replace("_", " ").lower()
                answer = f"{len(matched)} {readable_name} detected."
        else:
            matched = detections
            answer = f"Total of {len(matched)} objects detected."

        avg_conf = (sum(d["confidence"] for d in matched) / len(matched)) if matched else 0.0
        return {
            "answer": answer,
            "used_detection": True,
            "confidence_val": avg_conf,
            "supporting_detections": matched,
        }

    # 2. Presence check / Compliance check
    elif query_type == "presence_check":
        if target_class:
            neg_class = NEGATIVE_PAIRS.get(target_class, f"No_{target_class.lower()}")
            neg_matches = [d for d in detections if d["class"].lower() == neg_class.lower()]
            pos_matches = [d for d in detections if d["class"].lower() == target_class.lower()]

            if negated:
                # Asking about missing PPE (e.g. "Is anyone not wearing a helmet?")
                if len(neg_matches) > 0:
                    readable_target = target_class.replace("_", " ").lower()
                    if person_count > 0:
                        answer = f"Yes, {len(neg_matches)} of {person_count} person(s) detected is not wearing a {readable_target}."
                    else:
                        answer = f"Yes, {len(neg_matches)} violation(s) of missing {readable_target} detected."
                    supporting = neg_matches
                else:
                    readable_target = target_class.replace("_", " ").lower()
                    answer = f"No violations detected. No workers appear to be missing {readable_target}."
                    supporting = pos_matches
            else:
                # Asking about presence of PPE
                if len(pos_matches) > 0:
                    readable_target = target_class.replace("_", " ").lower()
                    answer = f"Yes, {len(pos_matches)} {readable_target} detected."
                    supporting = pos_matches
                else:
                    readable_target = target_class.replace("_", " ").lower()
                    answer = f"No, no {readable_target} detected in the image."
                    supporting = neg_matches

            avg_conf = (sum(d["confidence"] for d in supporting) / len(supporting)) if supporting else 0.0
            return {
                "answer": answer,
                "used_detection": True,
                "confidence_val": avg_conf,
                "supporting_detections": supporting,
            }

    # 3. Most common PPE item
    elif query_type == "most_common":
        ppe_detections = [d for d in detections if d["class"] != "Person"]
        if not ppe_detections:
            return {
                "answer": "No PPE items detected to determine frequency.",
                "used_detection": True,
                "confidence_val": 0.0,
                "supporting_detections": [],
            }

        counts = Counter(d["class"] for d in ppe_detections)
        most_common_class, freq = counts.most_common(1)[0]
        supporting = [d for d in ppe_detections if d["class"] == most_common_class]
        avg_conf = sum(d["confidence"] for d in supporting) / len(supporting)
        readable_name = most_common_class.replace("_", " ")
        answer = f"The most common item is '{readable_name}' with {freq} detection(s)."
        return {
            "answer": answer,
            "used_detection": True,
            "confidence_val": avg_conf,
            "supporting_detections": supporting,
        }

    # Fallback
    return {
        "answer": f"Processed query with {len(detections)} detection(s).",
        "used_detection": True,
        "confidence_val": 0.5,
        "supporting_detections": detections,
    }


# ── Stage 3: Confidence Guardrail ────────────────────────────────────────────

def apply_guardrail(
    reasoning_result: Dict[str, Any],
    all_detections: List[Dict[str, Any]],
    intent: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Stage 3: Confidence Guardrail.
    Flags insufficient information if:
    1. Confidence of supporting detections is strictly below the threshold.
    2. A target PPE class is queried but 0 relevant detections exist while workers are present.
    """
    if not reasoning_result.get("used_detection", True):
        return {
            "answer": reasoning_result["answer"],
            "used_detection": False,
            "confidence": "high",
            "supporting_detections": [],
        }

    conf_val = reasoning_result.get("confidence_val", 0.0)
    supporting = reasoning_result.get("supporting_detections", [])
    target_class = intent.get("target_class")
    person_count = sum(1 for d in all_detections if d["class"] == "Person")

    # Guardrail check 1: Low detection confidence
    if supporting and conf_val < threshold:
        return {
            "answer": "I can't confidently answer this from the detections — confidence too low.",
            "used_detection": True,
            "confidence": "low",
            "supporting_detections": supporting,
        }

    # Guardrail check 2: Target presupposed but completely absent in scene with people
    if target_class and target_class != "Person" and not supporting:
        if person_count > 0:
            return {
                "answer": f"I can't confidently answer this — no clear {target_class.replace('_', ' ').lower()} cues detected despite worker presence.",
                "used_detection": True,
                "confidence": "low",
                "supporting_detections": [],
            }

    confidence_level = "high" if conf_val >= 0.75 else "medium" if conf_val >= threshold else "low"

    return {
        "answer": reasoning_result["answer"],
        "used_detection": True,
        "confidence": confidence_level,
        "supporting_detections": supporting,
    }


def execute_pipeline(question: str, detections: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    """Execute complete 3-stage reasoning pipeline."""
    # Stage 1: Intent Routing
    intent = route_intent(question)

    # Stage 2: Deterministic Structured Reasoning
    reasoning_result = reason_over_detections(detections, intent)

    # Stage 3: Confidence Guardrail
    final_output = apply_guardrail(reasoning_result, detections, intent, threshold=threshold)
    return final_output
