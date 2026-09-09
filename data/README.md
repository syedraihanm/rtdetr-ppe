# PPE Detection Dataset Structure

This directory contains data preparation, inspection, and verification scripts for the 13-class Construction Site PPE dataset.

## Dataset Classes (13 Total)

| ID | Class Name | Category | Description |
|---|---|---|---|
| 0 | `Eye_protection` | PPE Compliant | Safety goggles / eye shields |
| 1 | `Foot_protection` | PPE Compliant | Steel-toe boots / safety shoes |
| 2 | `Hand_protection` | PPE Compliant | Safety gloves |
| 3 | `Head_protection` | PPE Compliant | Hardhat / helmet |
| 4 | `No_eye_protection` | Non-Compliant | Eyes exposed without protection |
| 5 | `No_foot_protection` | Non-Compliant | Unprotected footwear |
| 6 | `No_hand_protection` | Non-Compliant | Bare hands without gloves |
| 7 | `No_head_protection` | Non-Compliant | Bare head without helmet |
| 8 | `No_respiratory_protection` | Non-Compliant | Uncovered respiratory area |
| 9 | `No_safety_vest` | Non-Compliant | Worker without high-vis safety vest |
| 10 | `Respiratory_protection` | PPE Compliant | Dust mask / respirator |
| 11 | `Safety_vest` | PPE Compliant | High-visibility safety vest |
| 12 | `Person` | Context / Anchor | Full person body bounding box |

## Directory Layout (Local & Cloud)

Dataset images and annotations are structured according to standard YOLO format:

```
rtdetr-ppe/
├── data.yaml
├── train/
│   ├── images/  (*.jpg, *.png)
│   └── labels/  (*.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

> **Note:** Raw image files are excluded from Git tracking via `.gitignore` due to size constraints. Download the raw dataset from [Roboflow Universe](https://universe.roboflow.com/fahim-shahriar-2frao/construction-site-safety-v5wfl/dataset/2).

## Data Pipeline Scripts

1. **Auto-label Person class**:
   ```bash
   uv run python add_person_class.py --dataset-root .
   ```
2. **Sanitize Roboflow mixed segmentation lines**:
   ```bash
   uv run python fix_mixed_labels.py
   ```
3. **Verify dataset integrity & class balance**:
   ```bash
   uv run python data/prepare_split.py --check-integrity
   ```
