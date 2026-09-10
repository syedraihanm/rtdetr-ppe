# Model Weights Directory

This directory stores the fine-tuned RT-DETR-L model weights (`best.pt`, ~246 MB).

## Download Pre-Trained Weights (Recommended)

The fine-tuned checkpoint is hosted on GitHub Releases:

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

## Inference Fallback Order

`app/detection.py` resolves the model path in this order:
1. `MODEL_PATH` environment variable
2. `weights/best.pt`
3. `runs/train/ppe_rtdetr/weights/best.pt`
4. `rtdetr-l.pt` (base model fallback)
