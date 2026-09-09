"""
train.py  —  RT-DETR fine-tuning on the PPE dataset

Usage
-----
# 1. Local CPU smoke test (~10 images, 1 epoch) — catch config bugs before cloud
uv run train.py --smoke-test

# 2. Real training on Kaggle / Colab (paste this script + the dataset)
uv run train.py --model rtdetr-l.pt --epochs 100 --batch 16

# 3. Resume from a checkpoint
uv run train.py --model runs/train/ppe_rtdetr/weights/last.pt --epochs 100 --batch 16

Reproducibility
---------------
Every run appends a timestamped entry to TRAINING.md. The full Ultralytics
args.yaml is also saved automatically inside the run directory.

Environment
-----------
Tested with:
    ultralytics >= 8.4
    torch >= 2.0  (CPU for smoke test, CUDA for real runs)
"""

import argparse
import datetime
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def get_device_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu": "N/A",
        "gpu_count": 0,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def make_smoke_dataset(dataset_root: Path, n_images: int = 10) -> Path:
    """
    Copies a tiny subset of the train split to a temp folder and writes a
    temporary data.yaml pointing at it. Used for local CPU smoke tests only.
    """
    import shutil

    smoke_dir = dataset_root / "_smoke_test"
    smoke_dir.mkdir(exist_ok=True)

    # Load the full data.yaml to get class names
    with open(dataset_root / "data.yaml") as f:
        full_data = yaml.safe_load(f)

    for split in ("train", "valid"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        smoke_img_dir = smoke_dir / split / "images"
        smoke_lbl_dir = smoke_dir / split / "labels"
        smoke_img_dir.mkdir(parents=True, exist_ok=True)
        smoke_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        sample = images[:n_images]
        for img_path in sample:
            shutil.copy2(img_path, smoke_img_dir / img_path.name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, smoke_lbl_dir / lbl_path.name)
            else:
                # Create an empty label file so YOLO doesn't crash
                (smoke_lbl_dir / (img_path.stem + ".txt")).touch()

    # Write smoke data.yaml
    smoke_yaml_path = smoke_dir / "data.yaml"
    smoke_data = {
        "train": str((smoke_dir / "train" / "images").resolve()),
        "val": str((smoke_dir / "valid" / "images").resolve()),
        "nc": full_data["nc"],
        "names": full_data["names"],
    }
    with open(smoke_yaml_path, "w") as f:
        yaml.safe_dump(smoke_data, f, sort_keys=False)

    print(f"Smoke dataset prepared at {smoke_dir} ({n_images} images per split)")
    return smoke_yaml_path


def append_training_log(
    log_path: Path,
    run_name: str,
    model: str,
    hyperparams: dict,
    device_info: dict,
    results: dict,
    wall_clock_s: float,
    smoke_test: bool,
) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = str(datetime.timedelta(seconds=int(wall_clock_s)))
    lines = [
        f"\n---\n",
        f"## Run: `{run_name}` {'[SMOKE TEST]' if smoke_test else ''}",
        f"**Date:** {now}",
        f"**Wall-clock time:** {duration}",
        f"",
        f"### Hardware",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Platform | {device_info['platform']} |",
        f"| Python | {device_info['python']} |",
        f"| PyTorch | {device_info['torch']} |",
        f"| CUDA available | {device_info['cuda_available']} |",
        f"| CUDA version | {device_info['cuda_version']} |",
        f"| GPU | {device_info['gpu']} |",
        f"| GPU count | {device_info['gpu_count']} |",
        f"",
        f"### Hyperparameters",
        f"```yaml",
    ]
    for k, v in hyperparams.items():
        lines.append(f"{k}: {v}")
    lines += [
        f"```",
        f"",
        f"### Results",
        f"```json",
        json.dumps(results, indent=2),
        f"```",
        f"",
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Training log appended to {log_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETR on PPE dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("."),
        help="Root folder containing data.yaml",
    )
    parser.add_argument(
        "--model", type=str, default="rtdetr-l.pt",
        help=(
            "Pretrained checkpoint to fine-tune from. "
            "Options: rtdetr-l.pt, rtdetr-x.pt, rtdetr-r18.pt. "
            "Pass a last.pt to resume."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size. Use 8-16 on T4 for rtdetr-l. Auto-reduced for smoke test.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial LR (AdamW)")
    parser.add_argument(
        "--freeze", type=int, default=4,
        help=(
            "Number of backbone layers to freeze at the start. "
            "Ultralytics RT-DETR: freeze=N freezes the first N layers. "
            "Set 0 to unfreeze everything from the start."
        ),
    )
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="",
        help="Device to use: '' = auto (GPU if available, else CPU), '0', '0,1', 'cpu'",
    )
    parser.add_argument(
        "--project", type=str, default="runs/train",
        help="Directory to save training runs",
    )
    parser.add_argument(
        "--name", type=str, default="ppe_rtdetr",
        help="Run name. Ultralytics will auto-increment (ppe_rtdetr, ppe_rtdetr2, ...)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers (set to 0 on Windows if you hit multiprocessing errors)",
    )
    # Augmentation controls (RT-DETR/Ultralytics knobs)
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip prob (keep 0 for hardhat domain)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip prob")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees (keep 0 for hardhat domain)")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="Hue augmentation")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="Saturation augmentation")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="Value augmentation")
    # Special modes
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Local CPU smoke test: uses 10-image subset, 1 epoch, batch=2. "
            "Run this first on your laptop to catch config bugs before cloud."
        ),
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip appending to TRAINING.md",
    )
    args = parser.parse_args()

    # ── Override settings for smoke test ────────────────────────────────────
    if args.smoke_test:
        print("\n" + "=" * 60)
        print("SMOKE TEST MODE — 10 images, 1 epoch, CPU, batch=2")
        print("This will catch config errors. NOT a training run.")
        print("=" * 60 + "\n")
        args.epochs = 1
        args.batch = 2
        args.patience = 0  # no early stopping needed
        args.workers = 0   # Windows multiprocessing safe
        args.freeze = 0    # no need to freeze for 1-epoch test
        args.device = "cpu"

    set_seed(args.seed)
    device_info = get_device_info()

    # ── Resolve data.yaml (smoke test uses a temp yaml) ─────────────────────
    if args.smoke_test:
        data_yaml = make_smoke_dataset(args.dataset_root, n_images=10)
    else:
        data_yaml = args.dataset_root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {data_yaml}. "
                "Run from the dataset root or pass --dataset-root."
            )

    # ── Resolve device ───────────────────────────────────────────────────────
    device = args.device
    if device == "" and not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be extremely slow on CPU.")
        print("         Use --smoke-test for a local sanity check, then move to Kaggle/Colab.")
        device = "cpu"

    # ── Import Ultralytics (deferred so --help is always fast) ───────────────
    from ultralytics import RTDETR

    print(f"Loading model: {args.model}")
    model = RTDETR(args.model)

    hyperparams = {
        "model": args.model,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "optimizer": "AdamW",
        "cos_lr": True,
        "freeze": args.freeze,
        "patience": args.patience,
        "seed": args.seed,
        "device": device or "auto",
        "workers": args.workers,
        "mosaic": args.mosaic,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "degrees": args.degrees,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "project": args.project,
        "name": args.name,
        "smoke_test": args.smoke_test,
    }

    print("\nHyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    print()

    # ── Train ────────────────────────────────────────────────────────────────
    t_start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer="AdamW",
        cos_lr=True,
        freeze=args.freeze,
        patience=args.patience,
        seed=args.seed,
        device=device if device else None,
        workers=args.workers,
        mosaic=args.mosaic,
        flipud=args.flipud,
        fliplr=args.fliplr,
        degrees=args.degrees,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        project=args.project,
        name=args.name,
        exist_ok=False,   # Ultralytics auto-increments the name
        verbose=True,
    )
    wall_clock_s = time.time() - t_start

    # ── Extract results for logging ──────────────────────────────────────────
    results_dict = {}
    try:
        # Ultralytics Results object — extract what's available
        results_dict = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", -1)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", -1)),
            "precision": float(results.results_dict.get("metrics/precision(B)", -1)),
            "recall": float(results.results_dict.get("metrics/recall(B)", -1)),
            "best_epoch": int(getattr(results, "best_epoch", -1)),
        }
    except Exception as e:
        results_dict = {"note": f"Could not extract results: {e}"}

    # Print final results
    print("\n" + "=" * 60)
    if args.smoke_test:
        print("SMOKE TEST COMPLETE — no meaningful metrics expected at 1 epoch.")
        print("Config is valid. Now run the full training on Kaggle/Colab.")
    else:
        print("TRAINING COMPLETE")
        print(f"mAP@50:     {results_dict.get('mAP50', 'N/A')}")
        print(f"mAP@50-95:  {results_dict.get('mAP50-95', 'N/A')}")
        print(f"Precision:  {results_dict.get('precision', 'N/A')}")
        print(f"Recall:     {results_dict.get('recall', 'N/A')}")
        print(f"Best epoch: {results_dict.get('best_epoch', 'N/A')}")
    print(f"Wall-clock: {str(datetime.timedelta(seconds=int(wall_clock_s)))}")
    print("=" * 60 + "\n")

    # ── Write to TRAINING.md ─────────────────────────────────────────────────
    if not args.no_log:
        log_path = args.dataset_root / "TRAINING.md"
        if not log_path.exists():
            # Write header on first run
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("# TRAINING.md — Reproducibility Log\n\n")
                f.write(
                    "This file is auto-updated by `train.py`. "
                    "Each run appends an entry with hardware, hyperparameters, and results.\n"
                )
        run_name = args.name
        append_training_log(
            log_path=log_path,
            run_name=run_name,
            model=args.model,
            hyperparams=hyperparams,
            device_info=device_info,
            results=results_dict,
            wall_clock_s=wall_clock_s,
            smoke_test=args.smoke_test,
        )


if __name__ == "__main__":
    main()
