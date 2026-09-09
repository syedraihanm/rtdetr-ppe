# TRAINING.md — Reproducibility Log

This file is auto-updated by `train.py`. Each run appends an entry with hardware, hyperparameters, and results.

---

## Dataset & Preprocessing

- **Source**: Roboflow Universe — "Construction Site Safety" dataset v2
  - Workspace: `fahim-shahriar-2frao` / Project: `construction-site-safety-v5wfl`
  - License: CC BY 4.0
  - URL: https://universe.roboflow.com/fahim-shahriar-2frao/construction-site-safety-v5wfl/dataset/2
- **Original classes (12)**: `Eye_protection`, `Foot_protection`, `Hand_protection`, `Head_protection`, `No_eye_protection`, `No_foot_protection`, `No_hand_protection`, `No_head_protection`, `No_respiratory_protection`, `No_safety_vest`, `Respiratory_protection`, `Safety_vest`
- **Person class (class 12)**: Auto-labeled using `yolo11n.pt` (COCO-pretrained) at `conf=0.5` via `add_person_class.py`. Labels appended to existing PPE `.txt` files — original PPE annotations untouched.
- **Final dataset (13 classes)**: All 12 original PPE classes + `Person` as index 12.

### Person Auto-Labeling Statistics

| Split | Images | Images with Person | Person Boxes | Speed |
|---|---|---|---|---|
| train | 8,956 | 5,693 (63.6%) | 9,087 | ~10.3 img/s |
| valid | 1,279 | 768 (60.0%) | 1,240 | ~11.6 img/s |
| test | 640 | 416 (65.0%) | 691 | ~9.8 img/s |

> **Memo note**: Person labels are model-generated, not human-verified. Spot-checked 40 preview images (`_person_label_preview/`). Full-body workers detected reliably; occluded/edge-cropped workers occasionally missed at `conf=0.5`. This is disclosed in the memo as a methodology note.

### Auto-Labeling Methodology Decision
- `flipud=0.0` and `degrees=0.0` — vertical flip and rotation disabled intentionally. A rotated or upside-down hardhat/vest is not a valid augmentation for a construction-safety domain.
- `conf=0.5` threshold for Person detection — conservative enough to avoid false positives on machinery/vehicles while still catching >60% of workers across all splits.

---

## Label Fix — Mixed Segmentation Rows (`fix_mixed_labels.py`)

**Discovered during smoke test.** The Roboflow export contained label files mixing YOLO segmentation rows (>5 values per line) with detection rows (exactly 5 values). Ultralytics silently ignores these files with the warning `labels mix segment and detection rows`.

**Impact**: 4,990 files (46% of the dataset) were being silently dropped before this fix.

**Fix**: `fix_mixed_labels.py` strips segmentation rows, keeping only the `class cx cy w h` detection rows. The segmentation polygon annotations are not meaningful for our detection task.

| Split | Files fixed | Seg rows dropped | Det rows kept |
|---|---|---|---|
| train | 4,112 | 4,112 | 17,498 |
| valid | 577 | 577 | 2,602 |
| test | 301 | 301 | 1,429 |

**Smoke test validation**: Before fix → 9/10 sample images corrupt. After fix → 0/10 corrupt, all 10 images used.

> **Memo note**: This is a real data quality issue discovered mid-project — documents it as a legitimate "mid-project pivot" note per the brief.

---

## Model Choice

| Model | Params | Notes |
|---|---|---|
| `rtdetr-l.pt` | ~32M | **Primary** — best accuracy/speed tradeoff |
| `rtdetr-x.pt` | ~67M | Alternative if mAP is insufficient (slower) |
| `rtdetr-r18.pt` | ~20M | Fallback — faster epochs, ~2-4 mAP points lower |

---

## Smoke Test (Local CPU)

```bash
uv run train.py --smoke-test
```

Run this locally (i5-1335U / 16GB RAM / CPU) before submitting to Kaggle/Colab.
- 10-image subset, 1 epoch, batch=2, CPU — purpose is config validation only.
- Expect no meaningful metrics at 1 epoch; pass = no crash.

---

## Cloud Training (Kaggle / Colab)

```bash
# On Kaggle T4 — recommended
python train.py --model rtdetr-l.pt --epochs 100 --batch 16 --device 0

# Resume from checkpoint
python train.py --model runs/train/ppe_rtdetr/weights/last.pt --epochs 100 --batch 16
```

> **Note**: Set `--workers 0` on Windows if you encounter multiprocessing errors.

---

## Environment

<!-- This section is filled in by the smoke test and real training runs via train.py -->

See per-run entries below for exact Ultralytics / PyTorch / CUDA versions.

---

<!-- Per-run log entries are appended below by train.py automatically -->

---

## Run: `ppe_rtdetr` [SMOKE TEST]
**Date:** 2026-09-09 22:10:43
**Wall-clock time:** 0:00:35

### Hardware
| Key | Value |
|-----|-------|
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.6 |
| PyTorch | 2.14.0+cpu |
| CUDA available | False |
| CUDA version | N/A |
| GPU | N/A |
| GPU count | 0 |

### Hyperparameters
```yaml
model: rtdetr-l.pt
data: _smoke_test\data.yaml
epochs: 1
batch: 2
imgsz: 640
lr0: 0.0001
optimizer: AdamW
cos_lr: True
freeze: 0
patience: 0
seed: 42
device: cpu
workers: 0
mosaic: 1.0
flipud: 0.0
fliplr: 0.5
degrees: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
project: runs/train
name: ppe_rtdetr
smoke_test: True
```

### Results
```json
{
  "mAP50": 0.00875,
  "mAP50-95": 0.0036000000000000003,
  "precision": 0.33685313196457883,
  "recall": 0.11666666666666665,
  "best_epoch": -1
}
```


---

## Run: `ppe_rtdetr_postfix` [SMOKE TEST]
**Date:** 2026-09-09 22:13:45
**Wall-clock time:** 0:01:27

### Hardware
| Key | Value |
|-----|-------|
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.6 |
| PyTorch | 2.14.0+cpu |
| CUDA available | False |
| CUDA version | N/A |
| GPU | N/A |
| GPU count | 0 |

### Hyperparameters
```yaml
model: rtdetr-l.pt
data: _smoke_test\data.yaml
epochs: 1
batch: 2
imgsz: 640
lr0: 0.0001
optimizer: AdamW
cos_lr: True
freeze: 0
patience: 0
seed: 42
device: cpu
workers: 0
mosaic: 1.0
flipud: 0.0
fliplr: 0.5
degrees: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
project: runs/train
name: ppe_rtdetr_postfix
smoke_test: True
```

### Results
```json
{
  "mAP50": 0.08585689391852969,
  "mAP50-95": 0.07323513281179556,
  "precision": 0.34494480806020317,
  "recall": 0.1782312925170068,
  "best_epoch": -1
}
```


---

## Run: `ppe_rtdetr` [SMOKE TEST]
**Date:** 2026-09-09 22:35:52
**Wall-clock time:** 0:01:55

### Hardware
| Key | Value |
|-----|-------|
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.6 |
| PyTorch | 2.14.0+cpu |
| CUDA available | False |
| CUDA version | N/A |
| GPU | N/A |
| GPU count | 0 |

### Hyperparameters
```yaml
model: rtdetr-l.pt
data: _smoke_test\data.yaml
epochs: 1
batch: 2
imgsz: 640
lr0: 0.0001
optimizer: AdamW
cos_lr: True
freeze: 0
patience: 0
seed: 42
device: cpu
workers: 0
mosaic: 1.0
flipud: 0.0
fliplr: 0.5
degrees: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
project: runs/train
name: ppe_rtdetr
smoke_test: True
```

### Results
```json
{
  "mAP50": 0.08585689391852969,
  "mAP50-95": 0.07323513281179556,
  "precision": 0.34494480806020317,
  "recall": 0.1782312925170068,
  "best_epoch": -1
}
```


---

## Run: `ppe_rtdetr` [SMOKE TEST]
**Date:** 2026-09-09 22:38:19
**Wall-clock time:** 0:01:46

### Hardware
| Key | Value |
|-----|-------|
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.6 |
| PyTorch | 2.14.0+cpu |
| CUDA available | False |
| CUDA version | N/A |
| GPU | N/A |
| GPU count | 0 |

### Hyperparameters
```yaml
model: rtdetr-l.pt
data: _smoke_test\data.yaml
epochs: 1
batch: 2
imgsz: 640
lr0: 0.0001
optimizer: AdamW
cos_lr: True
freeze: 0
patience: 0
seed: 42
device: cpu
workers: 0
mosaic: 1.0
flipud: 0.0
fliplr: 0.5
degrees: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
project: runs/train
name: ppe_rtdetr
smoke_test: True
```

### Results
```json
{
  "mAP50": 0.08585689391852969,
  "mAP50-95": 0.07323513281179556,
  "precision": 0.34494480806020317,
  "recall": 0.1782312925170068,
  "best_epoch": -1
}
```

