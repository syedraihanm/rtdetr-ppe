"""
app/detection.py — RT-DETR Inference Engine

Loads fine-tuned Ultralytics RT-DETR model and performs object detection on
input images. Returns bounding boxes formatted in xyxy pixel coordinates adhering
strictly to Section 5 of the project brief.
"""

from io import BytesIO
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR

logger = logging.getLogger("rtdetr-ppe.detection")

# Global model cache
_MODEL: Optional[RTDETR] = None
_MODEL_PATH: Optional[Path] = None


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Find the best available model weights file."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path("weights/best.pt"),
        Path("runs/train/ppe_rtdetr/weights/best.pt"),
        Path("rtdetr-l.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback to downloading or using rtdetr-l.pt
    return Path("rtdetr-l.pt")


def get_model(model_path: Optional[str] = None) -> RTDETR:
    """Load or retrieve the cached RT-DETR model."""
    global _MODEL, _MODEL_PATH
    target_path = resolve_model_path(model_path)

    if _MODEL is None or _MODEL_PATH != target_path:
        logger.info(f"Loading RT-DETR model checkpoint from: {target_path}")
        _MODEL = RTDETR(str(target_path))
        _MODEL_PATH = target_path
    return _MODEL


def predict_from_bytes(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run detection on raw image bytes.

    Returns:
        {
            "detections": [
                {
                    "class": str,
                    "confidence": float,
                    "box": [x1, y1, x2, y2]   # pixel coordinates [xmin, ymin, xmax, ymax]
                }, ...
            ],
            "image_width": int,
            "image_height": int
        }
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image from provided bytes.")

    h, w = img_bgr.shape[:2]

    # Convert BGR to RGB for model inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = get_model(model_path)
    results = model.predict(
        source=img_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        res = results[0]
        boxes = res.boxes
        names = model.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(coord, 2) for coord in box.xyxy[0].tolist()]

            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "box": xyxy,
            })

    return {
        "detections": detections,
        "image_width": int(w),
        "image_height": int(h),
    }
