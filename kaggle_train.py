"""
kaggle_train.py — Run this inside a Kaggle Notebook (T4/P100 GPU)

Setup on Kaggle:
1. Create a new Notebook → enable GPU accelerator (T4 x2 or P100)
2. Upload your dataset folder as a Kaggle Dataset, OR use the Roboflow API:
   - Go to kaggle.com/datasets → New Dataset → upload the zip of your project
3. Paste this file into a Kaggle code cell, or upload as a .py and run:
       !python kaggle_train.py
4. After training, download weights/best.pt from the output panel.

Alternative — Roboflow API direct download (no upload needed):
    !pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_KEY")
    project = rf.workspace("fahim-shahriar-2frao").project("construction-site-safety-v5wfl")
    dataset = project.version(2).download("yolov8")

Reproducibility command used for real training run (logged in TRAINING.md):
    python train.py --model rtdetr-l.pt --epochs 100 --batch 16 --device 0 --seed 42
"""

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def append_training_log(
    log_path: Path,
    run_name: str,
    model: str,
    hyperparams: dict,
    device_info: dict,
    results: dict,
    wall_clock_s: float,
) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = str(datetime.timedelta(seconds=int(wall_clock_s)))
    lines = [
        f"\n---\n",
        f"## Run: `{run_name}` [KAGGLE REAL RUN]",
        f"**Date:** {now}",
        f"**Wall-clock time:** {duration}",
        f"",
        f"### Hardware",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Platform | {device_info.get('platform', 'N/A')} |",
        f"| Python | {device_info.get('python', 'N/A')} |",
        f"| PyTorch | {device_info.get('torch', 'N/A')} |",
        f"| CUDA available | {device_info.get('cuda_available', 'N/A')} |",
        f"| CUDA version | {device_info.get('cuda_version', 'N/A')} |",
        f"| GPU | {device_info.get('gpu', 'N/A')} |",
        f"| GPU count | {device_info.get('gpu_count', 0)} |",
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
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Training log appended to {log_path}")
    except Exception as e:
        print(f"Warning: Could not write to {log_path}: {e}")


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists()


def setup_environment():
    """Install/upgrade dependencies on Kaggle."""
    print("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "pyyaml"],
        check=True,
    )
    print("Done.")


def find_dataset_root() -> Path:
    """Try to auto-locate the dataset root on Kaggle or locally."""
    candidates = [
        Path("/kaggle/input"),          # Kaggle dataset input
        Path("/kaggle/working"),        # Kaggle working dir
        Path("."),                      # Local
    ]
    for base in candidates:
        # Look for data.yaml anywhere under this root
        for yaml_path in base.rglob("data.yaml"):
            if (yaml_path.parent / "train").exists():
                print(f"Found dataset at: {yaml_path.parent}")
                return yaml_path.parent
    raise FileNotFoundError(
        "Could not find dataset root with data.yaml + train/. "
        "Set DATASET_ROOT environment variable or upload the dataset."
    )


def main():
    # ── Environment setup ────────────────────────────────────────────────────
    if is_kaggle():
        setup_environment()

    dataset_root = Path(os.environ.get("DATASET_ROOT", "")) or find_dataset_root()
    data_yaml = dataset_root / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    # ── Import after install ─────────────────────────────────────────────────
    import torch
    from ultralytics import RTDETR

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {data_yaml}")

    # ── Training config ──────────────────────────────────────────────────────
    # These match the hyperparameters logged in TRAINING.md
    SEED = 42
    EPOCHS = 100
    BATCH = 16           # T4 16GB handles this for rtdetr-l; lower to 8 if OOM
    IMGSZ = 640
    LR0 = 1e-4
    FREEZE = 4           # Freeze first 4 backbone layers for first run
    PATIENCE = 30        # Early stopping
    MODEL = "rtdetr-l.pt"
    PROJECT = "/kaggle/working/runs" if is_kaggle() else "runs/train"
    NAME = "ppe_rtdetr"

    import random
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device_info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    hyperparams = {
        "model": MODEL,
        "data": str(data_yaml),
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "lr0": LR0,
        "optimizer": "AdamW",
        "cos_lr": True,
        "freeze": FREEZE,
        "patience": PATIENCE,
        "seed": SEED,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 2,
        "mosaic": 1.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "degrees": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "project": PROJECT,
        "name": NAME,
    }

    # ── Train ────────────────────────────────────────────────────────────────
    t_start = time.time()
    model = RTDETR(MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        optimizer="AdamW",
        cos_lr=True,
        freeze=FREEZE,
        patience=PATIENCE,
        seed=SEED,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        # Augmentation — rotation/vflip OFF for construction domain
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
    )
    wall_clock_s = time.time() - t_start

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    metrics = {}
    try:
        metrics = {
            "mAP50": results.results_dict.get("metrics/mAP50(B)", "N/A"),
            "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", "N/A"),
            "precision": results.results_dict.get("metrics/precision(B)", "N/A"),
            "recall": results.results_dict.get("metrics/recall(B)", "N/A"),
        }
        print(f"mAP@50:      {metrics['mAP50']}")
        print(f"mAP@50-95:   {metrics['mAP50-95']}")
        print(f"Precision:   {metrics['precision']}")
        print(f"Recall:      {metrics['recall']}")
    except Exception:
        pass
    weights_dir = Path(PROJECT) / NAME / "weights"
    print(f"\nBest weights: {weights_dir / 'best.pt'}")
    print(f"Last weights: {weights_dir / 'last.pt'}")
    print("=" * 60)

    # ── Log to TRAINING.md ───────────────────────────────────────────────────
    log_targets = [Path("TRAINING.md"), Path("/kaggle/working/TRAINING.md")]
    for lp in log_targets:
        if lp.parent.exists():
            append_training_log(
                log_path=lp,
                run_name=NAME,
                model=MODEL,
                hyperparams=hyperparams,
                device_info=device_info,
                results=metrics,
                wall_clock_s=wall_clock_s,
            )

    # On Kaggle, weights are auto-saved as output
    if is_kaggle():
        print("\nKaggle: Download 'best.pt' and 'TRAINING.md' from the Output panel.")


if __name__ == "__main__":
    main()
