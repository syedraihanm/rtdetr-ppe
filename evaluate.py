"""
evaluate.py — Comprehensive Evaluation Script for RT-DETR PPE Model

Runs Ultralytics COCO-style evaluation (.val()) on validation or test splits.
Extracts:
- Overall & per-class mAP50, mAP50-95, Precision, Recall.
- Confusion matrix analysis specifically for safety-critical pairs:
    * Head_protection vs No_head_protection
    * Safety_vest vs No_safety_vest
- Saves failure analysis outputs and summaries for TRAINING.md and memo.pdf.

Usage:
    uv run python evaluate.py --weights weights/best.pt --split test
    uv run python evaluate.py --weights rtdetr-l.pt --split valid --batch 8
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from ultralytics import RTDETR


CRITICAL_PAIRS = [
    ("Head_protection", "No_head_protection"),
    ("Safety_vest", "No_safety_vest"),
    ("Eye_protection", "No_eye_protection"),
]


def run_evaluation(
    weights: str,
    data_yaml: str = "data.yaml",
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    conf: float = 0.001,
    iou: float = 0.6,
):
    print(f"\n================================================================================")
    print(f"EVALUATING MODEL: {weights} on split: {split}")
    print(f"================================================================================\n")

    model = RTDETR(weights)

    # Run Ultralytics validation
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
    )

    names = model.names
    class_names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)

    # Aggregate Metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"mAP@50     : {metrics.box.map50:.4f}")
    print(f"mAP@50-95  : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    # Per-Class Table
    print("\n" + "=" * 80)
    print("PER-CLASS EVALUATION BREAKDOWN")
    print("=" * 80)
    headers = ["Class ID", "Class Name", "Precision", "Recall", "mAP@50", "mAP@50-95"]
    col_w = [10, 28, 12, 12, 12, 12]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, col_w)))
    print("-+-".join("-" * w for w in col_w))

    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    map50_per_class = metrics.box.all_ap[:, 0] if hasattr(metrics.box, "all_ap") and metrics.box.all_ap is not None else [0] * len(class_names)
    map_per_class = metrics.box.maps if hasattr(metrics.box, "maps") and metrics.box.maps is not None else [0] * len(class_names)

    for i, cname in enumerate(class_names):
        p_val = p_per_class[i] if i < len(p_per_class) else 0.0
        r_val = r_per_class[i] if i < len(r_per_class) else 0.0
        m50_val = map50_per_class[i] if i < len(map50_per_class) else 0.0
        m_val = map_per_class[i] if i < len(map_per_class) else 0.0

        row = [
            str(i),
            cname,
            f"{p_val:.4f}",
            f"{r_val:.4f}",
            f"{m50_val:.4f}",
            f"{m_val:.4f}",
        ]
        print(" | ".join(f"{val:<{w}}" for val, w in zip(row, col_w)))

    # Confusion Matrix Analysis for Critical Pairs
    print("\n" + "=" * 80)
    print("CRITICAL SAFETY COMPLIANCE PAIR CONFUSION ANALYSIS")
    print("=" * 80)

    cm = None
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix.matrix

    if cm is not None and isinstance(cm, np.ndarray):
        print("Detailed confusion extracted between paired positive & negative classes:\n")
        for pos_name, neg_name in CRITICAL_PAIRS:
            if pos_name in class_names and neg_name in class_names:
                pos_idx = class_names.index(pos_name)
                neg_idx = class_names.index(neg_name)

                # cm is shaped [num_classes + 1, num_classes + 1] (last is background)
                true_pos_pred_neg = cm[pos_idx, neg_idx] if pos_idx < cm.shape[0] and neg_idx < cm.shape[1] else 0
                true_neg_pred_pos = cm[neg_idx, pos_idx] if neg_idx < cm.shape[0] and pos_idx < cm.shape[1] else 0

                print(f"Pair: {pos_name} vs {neg_name}")
                print(f"  - Actual {pos_name} misclassified as {neg_name}: {int(true_pos_pred_neg)}")
                print(f"  - Actual {neg_name} misclassified as {pos_name}: {int(true_neg_pred_pos)} (DANGEROUS: false compliance)")
    else:
        print("Ultralytics confusion matrix plot saved in runs/val directory.")

    save_dir = getattr(metrics, "save_dir", "runs/val")
    print(f"\nEvaluation plots & artifacts saved to: {save_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR PPE model.")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "valid", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for mAP")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallback = Path("rtdetr-l.pt")
        if fallback.exists():
            print(f"Notice: '{args.weights}' not found. Falling back to '{fallback}'.")
            args.weights = str(fallback)
        else:
            print(f"Error: Weights file '{args.weights}' does not exist.", file=sys.stderr)
            sys.exit(1)

    split_name = "val" if args.split in ("val", "valid") else "test"
    run_evaluation(
        weights=args.weights,
        data_yaml=args.data,
        split=split_name,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
