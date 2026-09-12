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
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR

logger = logging.getLogger("rtdetr-ppe.detection")

DEFAULT_WEIGHTS_URL = "https://github.com/syedraihanm/rtdetr-ppe/releases/download/v1.0/best.pt"

# Global model cache
_MODEL: Optional[RTDETR] = None
_MODEL_PATH: Optional[Path] = None


def ensure_model_weights(
    target_path: Path = Path("weights/best.pt"),
    url: Optional[str] = None,
) -> Optional[Path]:
    """
    Ensure model weights exist locally. If missing or incomplete (< 1MB),
    automatically download the fine-tuned checkpoint from GitHub Releases.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    release_url = url or os.getenv("MODEL_URL", DEFAULT_WEIGHTS_URL)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")

    logger.info(f"Model weights missing at {target_path}. Downloading from {release_url}...")
    print(f"[RT-DETR PPE] Model weights missing. Downloading fine-tuned weights from GitHub Release...")
    print(f"               URL: {release_url}")
    print(f"               Destination: {target_path}")

    req = urllib.request.Request(
        release_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RT-DETR-PPE-Client"},
    )

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size and total_size.isdigit() else 0
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            last_pct = -1

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = int((downloaded / total_bytes) * 100)
                    if pct % 20 == 0 and pct != last_pct:
                        print(f"[RT-DETR PPE] Download progress: {pct}% ({downloaded // (1024 * 1024)} MB / {total_bytes // (1024 * 1024)} MB)")
                        last_pct = pct

        temp_path.replace(target_path)
        print(f"[RT-DETR PPE] Weights successfully downloaded to {target_path} ({target_path.stat().st_size} bytes).")
        logger.info(f"Successfully downloaded model weights to {target_path}")
        return target_path

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        print(f"[RT-DETR PPE] Warning: Could not auto-download weights from {release_url}: {exc}")
        logger.warning(f"Failed to auto-download model weights: {exc}")
        return None


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Find the best available model weights file, auto-downloading fine-tuned checkpoint if missing."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 1. Existing fine-tuned checkpoints
    candidates = [
        Path("weights/best.pt"),
        Path("runs/train/ppe_rtdetr/weights/best.pt"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c

    # 2. Attempt automatic download of fine-tuned weights from GitHub Release
    downloaded = ensure_model_weights(Path("weights/best.pt"))
    if downloaded and downloaded.exists():
        return downloaded

    # 3. Fallback to base model checkpoint if network is unavailable
    fallback = Path("rtdetr-l.pt")
    if fallback.exists():
        return fallback

    return fallback


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
