# Model Weights Directory

This directory stores the fine-tuned RT-DETR-L model weights (`best.pt`, ~246 MB, **79.19% mAP@50**).

## Automatic Download (Default)

**No manual download is required!** When you start the FastAPI server (`uvicorn app.main:app`) or run inference through `app/detection.py`, the system checks for `weights/best.pt`. If missing or incomplete, it automatically streams and verifies the checkpoint directly from GitHub Releases with chunked progress reporting and atomic validation.

---

## Manual Download (Optional / Offline)

If you are running in an air-gapped environment or wish to pre-fetch weights:

```bash
# Linux / macOS / Git Bash
curl -L https://github.com/syedraihanm/rtdetr-ppe/releases/download/v1.0/best.pt -o weights/best.pt

# Windows PowerShell
Invoke-WebRequest -Uri https://github.com/syedraihanm/rtdetr-ppe/releases/download/v1.0/best.pt -OutFile weights/best.pt
```

**Direct link:** https://github.com/syedraihanm/rtdetr-ppe/releases/download/v1.0/best.pt

---

## Reproduce From Scratch (Optional)

If you want to retrain from the base checkpoint instead of using the released weights:

1. Run `kaggle_train.py` on Kaggle with a T4 GPU (see `TRAINING.md` for exact steps).
2. Download `runs/train/ppe_rtdetr/weights/best.pt` from the Kaggle output tab.
3. Place it at `weights/best.pt`.

---

## Pretrained Base Checkpoint

```bash
# Ultralytics will auto-download rtdetr-l.pt on first run
uv run python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt')"
```

---

## Inference Resolution Order

`app/detection.py` resolves the model path in this order:
1. `custom_path` argument or `MODEL_PATH` environment variable
2. Local `weights/best.pt` or `runs/train/ppe_rtdetr/weights/best.pt` (if present and > 1 MB)
3. **Automated download**: fetch fine-tuned `best.pt` from GitHub Release to `weights/best.pt`
4. `rtdetr-l.pt` (base COCO model fallback if offline)
