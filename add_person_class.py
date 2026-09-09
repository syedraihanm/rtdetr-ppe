"""
add_person_class.py

Auto-labels the COCO 'person' class into an existing PPE YOLO dataset
(train/valid/test, each with images/ + labels/) using a COCO-pretrained
detector, and appends it as a new class in data.yaml.

Why: your PPE dataset has no 'Person' class of its own. Rather than running
a second model at inference time, we bake person detections into the same
YOLO-format dataset so your fine-tuned RT-DETR learns Person alongside the
PPE classes in a single model.

IMPORTANT: these are model-generated labels, not human-verified ground
truth. Spot-check a sample before trusting them (see --visualize below),
and say so explicitly in your memo — this is a legitimate methodology
note, not something to hide.

Usage:
    # 1. Dry run first (e.g., test split or all) — see how many boxes WOULD be added, writes nothing
    uv run add_person_class.py --dataset-root . --split test --dry-run
    uv run add_person_class.py --dataset-root . --dry-run

    # 2. Write the labels across splits
    uv run add_person_class.py --dataset-root .

    # 3. Spot-check a handful of images with the new boxes drawn
    uv run add_person_class.py --dataset-root . --visualize 20
"""

import argparse
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

from ultralytics import YOLO
import yaml

COCO_PERSON_CLASS_ID = 0  # 'person' in standard COCO class ordering


def load_data_yaml(dataset_root: Path) -> dict:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No data.yaml found at {yaml_path}. "
            "Point --dataset-root at the folder that contains it."
        )
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def save_data_yaml(dataset_root: Path, data: dict) -> None:
    yaml_path = dataset_root / "data.yaml"
    backup_path = dataset_root / "data.yaml.bak"
    if not backup_path.exists():
        yaml_path.rename(backup_path)
        print(f"Backed up original data.yaml -> {backup_path}")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Wrote updated data.yaml with {data['nc']} classes.")


def find_split_dirs(dataset_root: Path) -> Dict[str, Tuple[Path, Path]]:
    """Locate train/valid/test image+label folders, tolerant of naming."""
    splits = {}
    for split_name in ("train", "valid", "val", "test"):
        img_dir = dataset_root / split_name / "images"
        lbl_dir = dataset_root / split_name / "labels"
        if img_dir.exists() and lbl_dir.exists():
            key = "valid" if split_name == "val" else split_name
            splits[key] = (img_dir, lbl_dir)
    if not splits:
        raise FileNotFoundError(
            f"Could not find any <split>/images + <split>/labels under {dataset_root}"
        )
    return splits


def xyxy_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int):
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def process_split(
    split_name: str,
    img_dir: Path,
    lbl_dir: Path,
    model: YOLO,
    person_class_id: int,
    conf_threshold: float,
    dry_run: bool,
    batch_size: int = 32,
) -> dict:
    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    total_imgs = len(image_paths)
    stats = {"images": 0, "person_boxes_added": 0, "images_with_person": 0}
    start_time = time.time()

    print(f"\nProcessing split [{split_name}] ({total_imgs} images, batch_size={batch_size})...")

    for i in range(0, total_imgs, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            batch=len(batch_paths),
            conf=conf_threshold,
            classes=[COCO_PERSON_CLASS_ID],
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            stats["images"] += 1
            if len(result.boxes) == 0:
                continue

            img_h, img_w = result.orig_shape
            new_lines = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xc, yc, w, h = xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h)
                new_lines.append(f"{person_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if new_lines:
                label_path = lbl_dir / (img_path.stem + ".txt")
                # Avoid appending duplicate person annotations if re-run
                existing_lines = []
                if label_path.exists():
                    try:
                        existing_lines = [
                            line.strip()
                            for line in label_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                    except Exception:
                        existing_lines = []

                # Filter out lines if person class is already recorded
                lines_to_add = [
                    l for l in new_lines
                    if not any(el.startswith(f"{person_class_id} ") for el in existing_lines)
                ]

                if lines_to_add:
                    stats["person_boxes_added"] += len(lines_to_add)
                    stats["images_with_person"] += 1
                    if not dry_run:
                        prefix = ""
                        if label_path.exists() and label_path.stat().st_size > 0:
                            with open(label_path, "rb") as f_check:
                                f_check.seek(-1, 2)
                                if f_check.read(1) not in (b"\n", b"\r"):
                                    prefix = "\n"
                        with open(label_path, "a", encoding="utf-8") as f:
                            if prefix:
                                f.write(prefix)
                            for line in lines_to_add:
                                f.write(line + "\n")
                elif existing_lines and any(el.startswith(f"{person_class_id} ") for el in existing_lines):
                    # Already had person box recorded previously
                    stats["images_with_person"] += 1

        elapsed = time.time() - start_time
        processed = stats["images"]
        fps = processed / elapsed if elapsed > 0 else 0
        eta_sec = (total_imgs - processed) / fps if fps > 0 else 0
        eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"
        print(
            f"\r[{split_name}] {processed}/{total_imgs} ({(processed/total_imgs)*100:.1f}%) | "
            f"Persons: {stats['images_with_person']} img, {stats['person_boxes_added']} boxes | "
            f"{fps:.1f} img/s | ETA: {eta_str}",
            end="",
            flush=True,
        )

    print(
        f"\n[{split_name}] Done: {stats['images']} images | "
        f"{stats['images_with_person']} got person boxes | "
        f"{stats['person_boxes_added']} person boxes added in {time.time()-start_time:.1f}s"
    )
    return stats


def visualize_sample(dataset_root: Path, splits: dict, n: int, class_names: list):
    import cv2

    out_dir = dataset_root / "_person_label_preview"
    out_dir.mkdir(exist_ok=True)

    all_images = []
    for split_name, (img_dir, lbl_dir) in splits.items():
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                all_images.append((split_name, img_path, lbl_path))

    # Prioritize images that actually have labels to make spot-checking meaningful
    labeled_images = [
        item for item in all_images
        if item[2].exists() and item[2].stat().st_size > 0
    ]
    sample_pool = labeled_images if len(labeled_images) >= n else all_images

    sample = random.sample(sample_pool, min(n, len(sample_pool)))
    for split_name, img_path, lbl_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)

                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                is_person = (cls_name.lower() == "person")
                color = (0, 0, 255) if is_person else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    cls_name,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        out_path = out_dir / f"{split_name}_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    print(f"Wrote {len(sample)} preview images to {out_dir} — open them and check the red Person boxes.")


def main():
    parser = argparse.ArgumentParser(description="Auto-label Person class into PPE YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Folder containing data.yaml and train/valid/test/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="COCO-pretrained model used only to source Person boxes",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for accepting a person detection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for YOLO inference",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "valid", "test"],
        help="Which split to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts, don't write any label files",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="If >0, save N random images with boxes drawn for a manual sanity check",
    )
    args = parser.parse_args()

    data = load_data_yaml(args.dataset_root)
    class_names = data["names"]
    if isinstance(class_names, dict):  # Roboflow sometimes uses {0: 'x', 1: 'y', ...}
        class_names = [class_names[i] for i in range(len(class_names))]

    splits = find_split_dirs(args.dataset_root)
    if args.split != "all":
        if args.split not in splits:
            raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(splits.keys())}")
        splits = {args.split: splits[args.split]}

    if "Person" in class_names:
        print("data.yaml already has a 'Person' class.")
        if args.visualize > 0:
            visualize_sample(args.dataset_root, splits, args.visualize, class_names)
        return

    person_class_id = len(class_names)
    print(f"Found splits to process: {list(splits.keys())}")
    print(f"New 'Person' class will be id {person_class_id} (total classes: {person_class_id + 1})")
    print(f"Loading {args.model} (COCO-pretrained)...")
    model = YOLO(args.model)

    totals = {"images": 0, "person_boxes_added": 0, "images_with_person": 0}
    for split_name, (img_dir, lbl_dir) in splits.items():
        stats = process_split(
            split_name,
            img_dir,
            lbl_dir,
            model,
            person_class_id,
            args.conf,
            args.dry_run,
            args.batch_size,
        )
        for k in totals:
            totals[k] += stats[k]

    print(
        f"\nTOTAL: {totals['images']} images | "
        f"{totals['images_with_person']} with person boxes | "
        f"{totals['person_boxes_added']} person boxes added "
        f"({'DRY RUN — nothing written' if args.dry_run else 'written to label files'})"
    )

    if not args.dry_run:
        if args.split in ("all", "train"):
            new_names = list(class_names) + ["Person"]
            data["names"] = new_names
            data["nc"] = len(new_names)
            save_data_yaml(args.dataset_root, data)
        else:
            print(
                f"\nNOTE: data.yaml was NOT updated yet because only the '{args.split}' split was processed. "
                "Run on 'train' (or --split all) to complete dataset labeling and update data.yaml."
            )

    if args.visualize > 0:
        updated_names = list(class_names) + ["Person"]
        visualize_sample(args.dataset_root, splits, args.visualize, updated_names)


if __name__ == "__main__":
    main()
