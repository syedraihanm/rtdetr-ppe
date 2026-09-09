# Model Weights Directory

This directory stores the fine-tuned RT-DETR model weights.

## Expected Checkpoint

Place your trained weights checkpoint at:
```
weights/best.pt
```

## How to Obtain Weights

### 1. From Kaggle / Google Colab
After running `kaggle_train.py` on Kaggle with GPU acceleration:
1. Navigate to the notebook's output tab.
2. Download `runs/train/ppe_rtdetr/weights/best.pt`.
3. Save it to `weights/best.pt` in this repository.

### 2. Pretrained Base Checkpoint
If fine-tuning or running tests with the base checkpoint:
```bash
# Ultralytics will auto-download rtdetr-l.pt upon first run
uv run python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt')"
```

## Inference Fallback
The FastAPI application (`app/detection.py`) automatically looks for weights in this order:
1. Path specified in `MODEL_PATH` environment variable
2. `weights/best.pt`
3. `runs/train/ppe_rtdetr/weights/best.pt`
4. `rtdetr-l.pt` (base model fallback)
