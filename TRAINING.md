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

---

## Run: `ppe_rtdetr` [KAGGLE REAL TRAINING RUN]
**Date:** 2026-09-10
**Wall-clock time:** 09:40:31 (34,830.9s across 71 epochs)
**Status:** Stopped at epoch 71 due to Kaggle session time limit. Model reached peak convergence at **Epoch 65** (fitness: 0.52347).

### Hardware
| Key | Value |
|-----|-------|
| Platform | Linux (Kaggle Cloud Environment) |
| Python | 3.12.x |
| PyTorch | 2.5.1+cu124 |
| CUDA available | True |
| CUDA version | 12.4 |
| GPU | Tesla T4 |
| GPU count | 2 (T4 ×2, device 0 utilized) |
| GPU VRAM | 16 GB (Peak usage ~9.5 GB) |

### Hyperparameters
```yaml
task: detect
mode: train
model: rtdetr-l.pt
data: /kaggle/input/datasets/raihan87/rtdetr/data.yaml
epochs: 100
batch: 16
imgsz: 640
lr0: 0.0001
lrf: 0.01
optimizer: AdamW
cos_lr: true
freeze: 4
patience: 30
seed: 42
device: '0'
workers: 4
mosaic: 1.0
flipud: 0.0
fliplr: 0.5
degrees: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
project: /kaggle/working/runs
name: ppe_rtdetr
```

### Best Model Performance (Epoch 65 Checkpoint `weights/best.pt`)
```json
{
  "best_epoch": 65,
  "mAP50": 0.79902,
  "mAP50-95": 0.52347,
  "precision": 0.78404,
  "recall": 0.79490,
  "val/giou_loss": 0.41283,
  "val/cls_loss": 0.61456,
  "val/l1_loss": 0.16320,
  "fitness": 0.52347
}
```

### Training Progression & Convergence Milestones
- **Epoch 1**: mAP@50 = 0.0401, mAP@50-95 = 0.0211, P = 0.1655, R = 0.4820
- **Epoch 5**: mAP@50 = 0.5955, mAP@50-95 = 0.3359, P = 0.5947, R = 0.6301
- **Epoch 10**: mAP@50 = 0.7084, mAP@50-95 = 0.4125, P = 0.6736, R = 0.7553
- **Epoch 20**: mAP@50 = 0.7718, mAP@50-95 = 0.4758, P = 0.7265, R = 0.7911
- **Epoch 30**: mAP@50 = 0.7890, mAP@50-95 = 0.4963, P = 0.7572, R = 0.7954
- **Epoch 45**: mAP@50 = 0.7977, mAP@50-95 = 0.5116, P = 0.7676, R = 0.8002
- **Epoch 60**: mAP@50 = **0.79955** (79.96%), mAP@50-95 = 0.52086, P = 0.77828, R = 0.79978 (Peak mAP@50)
- **Epoch 65 (Best Checkpoint)**: mAP@50 = 0.79902, mAP@50-95 = **0.52347** (52.35%), P = **0.78404**, R = 0.79490 (Peak Overall Fitness)
- **Epoch 71 (Final Run)**: mAP@50 = 0.79628, mAP@50-95 = 0.52014, P = 0.77082, R = 0.79971

> **Convergence Note**: By epoch 60, training curves reached an asymptotic plateau: cls_loss dropped from 1.237 down to 0.395, and mAP@50 stabilized right at 80.0%. The saved `weights/best.pt` file captures the peak fitness checkpoint at Epoch 65.

---

## Test Split Evaluation (`evaluate.py`)

Run on the held-out **640 test images** (2,233 annotated instances) using `weights/best.pt`:

```bash
uv run python evaluate.py --weights weights/best.pt --split test
```

### Overall Test Metrics
| Metric | Value |
|---|---|
| **mAP@50** | **0.7919 (79.19%)** |
| **mAP@50-95** | **0.5099 (50.99%)** |
| **Precision** | **0.7805 (78.05%)** |
| **Recall** | **0.7880 (78.80%)** |

### Per-Class Test Breakdown
| Class ID | Class Name | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| 0 | Eye_protection | 0.7066 | 0.8621 | 0.8789 | 0.5769 |
| 1 | Foot_protection | 0.8087 | 0.9388 | 0.9496 | 0.7413 |
| 2 | Hand_protection | 0.8695 | 0.8000 | 0.8567 | 0.4127 |
| 3 | Head_protection | 0.8619 | 0.8408 | 0.8027 | 0.5147 |
| 4 | No_eye_protection | 0.6697 | 0.8413 | 0.7739 | 0.4811 |
| 5 | No_foot_protection | 0.8426 | 0.8966 | 0.9193 | 0.6968 |
| 6 | No_hand_protection | 0.7688 | 0.6814 | 0.7150 | 0.2983 |
| 7 | No_head_protection | 0.8121 | 0.8182 | 0.8282 | 0.4956 |
| 8 | No_respiratory_protection | 0.8100 | 0.6986 | 0.7428 | 0.3973 |
| 9 | No_safety_vest | 0.7799 | 0.6206 | 0.6301 | 0.3951 |
| 10 | Respiratory_protection | 0.8812 | 0.8646 | 0.8663 | 0.6334 |
| 11 | Safety_vest | 0.7667 | 0.7709 | 0.7492 | 0.4998 |
| 12 | Person | 0.5690 | 0.6103 | 0.5823 | 0.4862 |

### Critical Safety Compliance Pair Confusion Analysis
| Safety Pair | Actual Positive → Pred Negative (False Violation) | Actual Negative → Pred Positive (**False Compliance - Critical**) |
|---|---|---|
| **Head_protection vs No_head_protection** | 15 instances | **41 instances** |
| **Safety_vest vs No_safety_vest** | 18 instances | **41 instances** |
| **Eye_protection vs No_eye_protection** | 6 instances | **3 instances** |

> **Memo Finding**: The confusion analysis exposes that **False Compliance** (a worker without PPE being classified as wearing PPE) occurs more frequently than False Violations. This insight justifies Stage 3 in our reasoning pipeline (`app/reasoning.py`), where low-confidence detections trigger an explicit conservative warning rather than falsely clearing a worker.



