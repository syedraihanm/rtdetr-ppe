"""
app/main.py — FastAPI Application for PPE Detection & Reasoning

Exposes:
- POST /detect : Object detection returning xyxy pixel bounding boxes and classes.
- POST /ask    : Multi-modal question answering using the 3-stage reasoning layer.
- GET  /health : Health check endpoint.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.detection import predict_from_bytes
from app.reasoning import apply_guardrail, execute_pipeline, reason_over_detections, route_intent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtdetr-ppe.api")

app = FastAPI(
    title="Construction Site PPE Detection & Reasoning API",
    description="Ultralytics RT-DETR PPE detection with a hand-written 3-stage reasoning decision layer.",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Response Models ─────────────────────────────────────────────────

class DetectionItem(BaseModel):
    class_: str = Field(alias="class", description="Detected class name")
    confidence: float = Field(..., description="Detection confidence score (0.0 - 1.0)")
    box: List[float] = Field(..., description="Bounding box in xyxy pixel coordinates: [x1, y1, x2, y2]")

    class Config:
        populate_by_name = True


class DetectResponse(BaseModel):
    detections: List[DetectionItem]
    image_width: int
    image_height: int


class AskResponse(BaseModel):
    answer: str
    used_detection: bool
    confidence: str
    supporting_detections: List[Dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health and readiness check."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/detect", response_model=DetectResponse, tags=["Inference"])
async def detect_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG/PNG) to analyze"),
    conf: float = 0.25,
):
    """
    Run RT-DETR object detection on an uploaded construction site image.
    Returns detected PPE items and people with [x1, y1, x2, y2] bounding boxes in pixel coordinates.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image, got: {file.content_type}",
        )

    try:
        contents = await file.read()
        results = predict_from_bytes(contents, conf_threshold=conf)
        return results
    except Exception as e:
        logger.error(f"Error during detection inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )


@app.post("/ask", response_model=AskResponse, tags=["Reasoning"])
async def ask_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG/PNG) for context"),
    question: str = Form(..., description="Natural language question about the site or PPE compliance"),
    conf: float = 0.25,
    guardrail_threshold: float = 0.5,
):
    """
    Query the construction site image with natural language.
    Executes a 3-stage hand-written reasoning pipeline:
    1. Intent Router (slot-extracts intent via raw LLM HTTP call).
    2. Structured Reasoning (deterministic Python business logic).
    3. Confidence Guardrail (returns 'insufficient information' if evidence is weak).
    """
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image, got: {file.content_type}",
        )

    try:
        # Stage 1: Intent Routing — decide whether query needs detector invocation
        intent = route_intent(question)

        detections = []
        if intent.get("needs_detection", True):
            contents = await file.read()
            det_output = predict_from_bytes(contents, conf_threshold=conf)
            detections = det_output.get("detections", [])

        # Stage 2: Structured Reasoning over detected objects
        reasoning_result = reason_over_detections(detections, intent)

        # Stage 3: Confidence Guardrail validation
        pipeline_result = apply_guardrail(
            reasoning_result,
            all_detections=detections,
            intent=intent,
            threshold=guardrail_threshold,
        )
        return pipeline_result
    except Exception as e:
        logger.error(f"Error during reasoning pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reasoning query failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
