"""
verify_reasoning.py — Unit tests for the reasoning pipeline (no model loading needed).
Tests Stage 1 (rule-based fallback), Stage 2 (deterministic logic), Stage 3 (guardrail).
"""

from app.reasoning import (
    _rule_based_fallback_intent,
    reason_over_detections,
    apply_guardrail,
    normalize_class_name,
    execute_pipeline,
)


PASS = 0
FAIL = 0

def check(label: str, condition: bool, got=None):
    global PASS, FAIL
    if condition:
        print(f"  [OK]   {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}" + (f" — got: {got}" if got is not None else ""))
        FAIL += 1


print("=" * 60)
print("TEST: normalize_class_name")
print("=" * 60)
check("hardhat -> Head_protection", normalize_class_name("hardhat") == "Head_protection")
check("vest -> Safety_vest",        normalize_class_name("vest") == "Safety_vest")
check("goggles -> Eye_protection",  normalize_class_name("goggles") == "Eye_protection")
check("mask -> Respiratory_protection", normalize_class_name("mask") == "Respiratory_protection")
check("PERSON (uppercase) -> normalized to Person", normalize_class_name("PERSON") == "Person")  # lowercased 'person' is in alias map
check("None input -> None",         normalize_class_name(None) is None)


print()
print("=" * 60)
print("TEST: Stage 1 — Rule-Based Fallback Intent Router")
print("=" * 60)

i = _rule_based_fallback_intent("Is anyone not wearing a helmet?")
check("query_type = presence_check",  i["query_type"] == "presence_check", i["query_type"])
check("target_class = Head_protection", i["target_class"] == "Head_protection", i["target_class"])
check("negated = True",               i["negated"] == True)
check("needs_detection = True",       i["needs_detection"] == True)

i = _rule_based_fallback_intent("How many workers are wearing vests?")
check("count query for vest",         i["query_type"] == "count", i["query_type"])
check("target = Safety_vest",         i["target_class"] == "Safety_vest", i["target_class"])

i = _rule_based_fallback_intent("What PPE is most common?")
check("most_common query",            i["query_type"] == "most_common", i["query_type"])

i = _rule_based_fallback_intent("What is the capital of France?")
check("off-topic -> no detection",    i["needs_detection"] == False, i)
check("off-topic query_type",         i["query_type"] == "general_no_detection_needed", i["query_type"])

# Edge case: "Is everyone wearing a vest?" (universal compliance check)
i = _rule_based_fallback_intent("Is everyone wearing a vest?")
check("everyone vest -> presence_check",   i["query_type"] == "presence_check", i["query_type"])
check("everyone vest -> target Safety_vest", i["target_class"] == "Safety_vest", i["target_class"])
check("everyone vest -> negated = True (checks non-compliance)", i["negated"] == True, i["negated"])
check("everyone vest -> needs_detection = True", i["needs_detection"] == True)


print()
print("=" * 60)
print("TEST: Stage 2 — Deterministic Reasoning")
print("=" * 60)

SAMPLE_DETECTIONS = [
    {"class": "Person",             "confidence": 0.95, "box": [10, 10, 200, 400]},
    {"class": "Person",             "confidence": 0.92, "box": [210, 10, 400, 400]},
    {"class": "Head_protection",    "confidence": 0.88, "box": [30, 20, 80, 60]},
    {"class": "No_head_protection", "confidence": 0.76, "box": [220, 20, 270, 60]},
    {"class": "Safety_vest",        "confidence": 0.81, "box": [30, 100, 200, 350]},
    {"class": "No_safety_vest",     "confidence": 0.70, "box": [210, 100, 400, 350]},
    {"class": "Safety_vest",        "confidence": 0.79, "box": [410, 10, 600, 400]},
]

# Test: count all persons
r = reason_over_detections(SAMPLE_DETECTIONS, {"needs_detection": True, "query_type": "count", "target_class": "Person", "negated": False})
check("count Person = 2",            "2" in r["answer"], r["answer"])
check("supporting has 2 detections", len(r["supporting_detections"]) == 2, len(r["supporting_detections"]))

# Test: presence_check — negated (missing helmet)
r = reason_over_detections(SAMPLE_DETECTIONS, {"needs_detection": True, "query_type": "presence_check", "target_class": "Head_protection", "negated": True})
check("missing helmet detected",     "1" in r["answer"], r["answer"])
check("supporting = No_head_protection", r["supporting_detections"][0]["class"] == "No_head_protection")

# Test: presence_check — NOT negated (wearing vest)
r = reason_over_detections(SAMPLE_DETECTIONS, {"needs_detection": True, "query_type": "presence_check", "target_class": "Safety_vest", "negated": False})
check("vest present",                "Yes" in r["answer"], r["answer"])
check("2 vest detections",           len(r["supporting_detections"]) == 2, len(r["supporting_detections"]))

# Test: most common (Safety_vest appears twice, others once)
r = reason_over_detections(SAMPLE_DETECTIONS, {"needs_detection": True, "query_type": "most_common", "target_class": None, "negated": False})
check("most common = Safety_vest",   "Safety_vest" in r["answer"] or "Safety vest" in r["answer"], r["answer"])

# Test: off-topic (needs_detection=False)
r = reason_over_detections(SAMPLE_DETECTIONS, {"needs_detection": False, "query_type": "general_no_detection_needed", "target_class": None, "negated": False})
check("no detection used",           r["used_detection"] == False, r)


print()
print("=" * 60)
print("TEST: Stage 3 — Confidence Guardrail")
print("=" * 60)

# Guardrail: low confidence should trigger insufficient-info
low_conf_result = {
    "answer": "Yes, 1 worker is missing helmet.",
    "used_detection": True,
    "confidence_val": 0.30,  # below 0.5
    "supporting_detections": [{"class": "No_head_protection", "confidence": 0.30, "box": [1,2,3,4]}],
}
intent_hm = {"needs_detection": True, "query_type": "presence_check", "target_class": "Head_protection", "negated": True}
g = apply_guardrail(low_conf_result, SAMPLE_DETECTIONS, intent_hm, threshold=0.5)
check("guardrail fires on conf < 0.5",       g["confidence"] == "low", g["confidence"])
check("guardrail message correct",           "can't confidently" in g["answer"].lower(), g["answer"])

# Guardrail: high confidence passes through
hi_conf_result = {
    "answer": "Yes, 1 worker is missing helmet.",
    "used_detection": True,
    "confidence_val": 0.82,
    "supporting_detections": [{"class": "No_head_protection", "confidence": 0.82, "box": [1,2,3,4]}],
}
g = apply_guardrail(hi_conf_result, SAMPLE_DETECTIONS, intent_hm, threshold=0.5)
check("guardrail passes on conf >= 0.75 (high)",  g["confidence"] == "high", g["confidence"])
check("answer preserved on pass-through",          g["answer"] == hi_conf_result["answer"], g["answer"])

# Guardrail: no supporting detections but workers present -> insufficient
empty_result = {
    "answer": "No violations detected. No workers appear to be missing head protection.",
    "used_detection": True,
    "confidence_val": 0.0,
    "supporting_detections": [],
}
g = apply_guardrail(empty_result, SAMPLE_DETECTIONS, intent_hm, threshold=0.5)
check("guardrail fires: target absent, workers present", g["confidence"] == "low", g["confidence"])


print()
print("=" * 60)
print("TEST: Full pipeline integration (no LLM — rule-based fallback)")
print("=" * 60)

out = execute_pipeline("Is anyone not wearing a helmet?", SAMPLE_DETECTIONS, threshold=0.5)
check("pipeline returns answer field",          "answer" in out, out)
check("pipeline returns confidence field",      "confidence" in out, out)
check("pipeline returns used_detection field",  "used_detection" in out, out)
check("pipeline answer mentions missing",       any(w in out["answer"].lower() for w in ["not", "missing", "without", "violation"]), out["answer"])
check("pipeline copy polish: 'missing head protection violation'", "missing head protection violation" in out["answer"], out["answer"])
print(f"  Full pipeline output: {out['answer']!r} (confidence={out['confidence']})")

# Safety-critical edge case: "Is everyone wearing a vest?"
# SAMPLE_DETECTIONS has 2 Safety_vest and 1 No_safety_vest.
# Fallback router must NOT treat it as normal positive-presence (which would see 2 vests and falsely say "Yes").
# It must flag the No_safety_vest violation!
out_vest = execute_pipeline("Is everyone wearing a vest?", SAMPLE_DETECTIONS, threshold=0.5)
check("everyone vest -> flags violation", "violation" in out_vest["answer"].lower(), out_vest["answer"])
check("everyone vest -> 1 missing safety vest violation detected", "1 missing safety vest violation detected" in out_vest["answer"], out_vest["answer"])
check("everyone vest -> supporting is No_safety_vest", out_vest["supporting_detections"][0]["class"] == "No_safety_vest")
print(f"  Pipeline 'Is everyone wearing a vest?' output: {out_vest['answer']!r} (supporting={out_vest['supporting_detections'][0]['class']})")


print()
print("=" * 60)
print(f"SUMMARY: {PASS} passed / {PASS + FAIL} total  ({'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'})")
print("=" * 60)

if FAIL > 0:
    raise SystemExit(1)
