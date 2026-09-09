"""
data/prepare_split.py — Dataset Split Verification & Class Balance Analysis

Inspects the PPE dataset splits (train, valid, test), counts instance annotations
per class, verifies annotation integrity (5-element bounding boxes in normalized coordinates),
and prints class distribution and balance metrics required for TRAINING.md and project memo.

Usage:
    uv run python data/prepare_split.py
    uv run python data/prepare_split.py --dataset-root . --markdown
    uv run python data/prepare_split.py --check-integrity
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import yaml


DEFAULT_CLASSES = [
    "Eye_protection",
    "Foot_protection",
    "Hand_protection",
    "Head_protection",
    "No_eye_protection",
    "No_foot_protection",
    "No_hand_protection",
    "No_head_protection",
    "No_respiratory_protection",
    "No_safety_vest",
    "Respiratory_protection",
    "Safety_vest",
    "Person",
]


def load_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, "r") as f:
            meta = yaml.safe_load(f)
            if "names" in meta:
                if isinstance(meta["names"], list):
                    return meta["names"]
                elif isinstance(meta["names"], dict):
                    return [meta["names"][k] for k in sorted(meta["names"].keys(), key=int)]
    return DEFAULT_CLASSES


def find_split_dir(dataset_root: Path, split_name: str) -> Path | None:
    candidates = [
        dataset_root / split_name,
        dataset_root / ("val" if split_name == "valid" else split_name),
        dataset_root / ("valid" if split_name == "val" else split_name),
    ]
    for c in candidates:
        if c.exists() and (c / "labels").exists():
            return c
    return None


def analyze_split(split_dir: Path, class_names: list[str], check_integrity: bool = False):
    lbl_dir = split_dir / "labels"
    img_dir = split_dir / "images"

    label_files = sorted(lbl_dir.glob("*.txt"))
    img_files = list(img_dir.glob("*.*")) if img_dir.exists() else []

    class_counts = Counter()
    image_has_class = defaultdict(set)
    corrupt_files = []
    total_boxes = 0

    for lbl_path in label_files:
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                corrupt_files.append((lbl_path.name, line_num, f"Expected 5 values (det), got {len(parts)}"))
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                corrupt_files.append((lbl_path.name, line_num, "Non-numeric values in line"))
                continue

            if cls_id < 0 or cls_id >= len(class_names):
                corrupt_files.append((lbl_path.name, line_num, f"Class id {cls_id} out of bounds (0..{len(class_names)-1})"))
                continue

            if check_integrity:
                for c in coords:
                    if c < -0.05 or c > 1.05:
                        corrupt_files.append((lbl_path.name, line_num, f"Coord out of [0, 1] range: {coords}"))
                        break

            class_name = class_names[cls_id]
            class_counts[class_name] += 1
            image_has_class[class_name].add(lbl_path.stem)
            total_boxes += 1

    return {
        "split_name": split_dir.name,
        "image_count": len(img_files),
        "label_count": len(label_files),
        "total_boxes": total_boxes,
        "class_counts": class_counts,
        "image_has_class": {k: len(v) for k, v in image_has_class.items()},
        "corrupt_files": corrupt_files,
    }


def print_summary(stats_by_split: dict, class_names: list[str], as_markdown: bool = False):
    splits = list(stats_by_split.keys())
    print("\n" + "=" * 80)
    print("DATASET SPLIT & CLASS DISTRIBUTION REPORT")
    print("=" * 80 + "\n")

    # Table 1: Splits Overview
    print("### 1. Split Counts")
    print("-" * 50)
    print(f"{'Split':<12} {'Images':<12} {'Labels':<12} {'Total Boxes':<12}")
    print("-" * 50)
    total_imgs = 0
    total_lbls = 0
    total_boxes = 0
    for split, s in stats_by_split.items():
        print(f"{split:<12} {s['image_count']:<12} {s['label_count']:<12} {s['total_boxes']:<12}")
        total_imgs += s['image_count']
        total_lbls += s['label_count']
        total_boxes += s['total_boxes']
    print("-" * 50)
    print(f"{'Total':<12} {total_imgs:<12} {total_lbls:<12} {total_boxes:<12}\n")

    # Table 2: Per-Class Counts
    print("### 2. Class Instance Counts Across Splits")
    headers = ["Class ID", "Class Name"] + [f"{s} (boxes)" for s in splits] + ["Total Boxes"]
    col_w = [10, 28] + [16 for _ in splits] + [14]
    hdr_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
    sep_line = "-+-".join("-" * w for w in col_w)

    print(hdr_line)
    print(sep_line)
    for idx, cname in enumerate(class_names):
        row_counts = [stats_by_split[s]["class_counts"].get(cname, 0) for s in splits]
        row_tot = sum(row_counts)
        vals = [f"{idx}", cname] + [str(c) for c in row_counts] + [str(row_tot)]
        print(" | ".join(f"{v:<{w}}" for v, w in zip(vals, col_w)))

    # Table 3: Safety Compliance Pair Balance
    print("\n### 3. Key PPE Compliance Pair Balance (Overall)")
    print("-" * 60)
    pairs = [
        ("Head_protection", "No_head_protection"),
        ("Safety_vest", "No_safety_vest"),
        ("Eye_protection", "No_eye_protection"),
        ("Foot_protection", "No_foot_protection"),
        ("Hand_protection", "No_hand_protection"),
    ]
    for pos, neg in pairs:
        pos_cnt = sum(stats_by_split[s]["class_counts"].get(pos, 0) for s in splits)
        neg_cnt = sum(stats_by_split[s]["class_counts"].get(neg, 0) for s in splits)
        ratio = (pos_cnt / neg_cnt) if neg_cnt > 0 else 0
        print(f"{pos:<22} : {pos_cnt:<6} | {neg:<22} : {neg_cnt:<6} (Positive:Negative = {ratio:.2f}:1)")

    # Integrity summary
    print("\n### 4. Integrity Check")
    all_corrupt = sum(len(s["corrupt_files"]) for s in stats_by_split.values())
    if all_corrupt == 0:
        print(" [OK] All label files conform to 5-element normalized detection format (0 corrupt lines).")
    else:
        print(f" [WARNING] Found {all_corrupt} corrupt lines across splits!")
        for s, data in stats_by_split.items():
            if data["corrupt_files"]:
                print(f"  - Split '{s}': {len(data['corrupt_files'])} corrupt issues")


def fix_split_concatenations(split_dir: Path) -> int:
    lbl_dir = split_dir / "labels"
    fixed_count = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text()
        new_lines = []
        modified = False
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) == 9 and parts[4].endswith("12"):
                # Missing newline between previous height and class 12 (Person)
                line1 = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4][:-2]}"
                line2 = f"12 {parts[5]} {parts[6]} {parts[7]} {parts[8]}"
                new_lines.append(line1)
                new_lines.append(line2)
                modified = True
                fixed_count += 1
            else:
                new_lines.append(line_str)
        if modified:
            lbl_path.write_text("\n".join(new_lines) + "\n")
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="Inspect and verify PPE dataset splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="Path to dataset root")
    parser.add_argument("--check-integrity", action="store_true", help="Perform coordinate bounds check")
    parser.add_argument("--fix", action="store_true", help="Automatically fix missing newline concatenated lines")
    parser.add_argument("--markdown", action="store_true", help="Output summary in markdown format")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    class_names = load_class_names(dataset_root)

    if args.fix:
        print("Running automatic label concatenation fix...")
        total_fixed = 0
        for split in ["train", "valid", "test"]:
            s_dir = find_split_dir(dataset_root, split)
            if s_dir:
                fixed = fix_split_concatenations(s_dir)
                print(f"  Fixed {fixed} concatenated lines in '{split}' split.")
                total_fixed += fixed
        print(f"Done! Total fixed: {total_fixed}\n")

    stats = {}
    for split in ["train", "valid", "test"]:
        s_dir = find_split_dir(dataset_root, split)
        if s_dir:
            stats[split] = analyze_split(s_dir, class_names, check_integrity=args.check_integrity)
        else:
            print(f"Warning: Split '{split}' directory not found at {dataset_root}", file=sys.stderr)

    if not stats:
        print("No valid dataset splits found.", file=sys.stderr)
        sys.exit(1)

    print_summary(stats, class_names, as_markdown=args.markdown)


if __name__ == "__main__":
    main()
