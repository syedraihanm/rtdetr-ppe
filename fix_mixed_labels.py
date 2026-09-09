"""
fix_mixed_labels.py

The Roboflow export contains labels that mix YOLO segmentation rows (>5 values)
with detection rows (exactly 5 values: class cx cy w h). Ultralytics silently
ignores these files entirely with the warning:
    "ignoring corrupt image/label: labels mix segment and detection rows"

This script strips the segmentation rows, keeping ONLY the detection rows
(class cx cy w h). Segmentation polygons from a 2D bird's-eye construction
dataset aren't meaningful for our detection task anyway.

Usage:
    # Dry run first — shows counts, writes nothing
    uv run fix_mixed_labels.py --dry-run

    # Fix for real
    uv run fix_mixed_labels.py

    # Fix a single split
    uv run fix_mixed_labels.py --split train
"""

import argparse
from pathlib import Path


def audit_label(path: Path):
    """Returns (has_det, has_seg, lines)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False, False, []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False, False, []
    has_det = any(len(l.split()) == 5 for l in lines)
    has_seg = any(len(l.split()) > 5 for l in lines)
    return has_det, has_seg, lines


def fix_split(split_dir: Path, dry_run: bool) -> dict:
    lbl_dir = split_dir / "labels"
    stats = {"total": 0, "mixed": 0, "fixed": 0, "seg_rows_dropped": 0, "det_rows_kept": 0}

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        stats["total"] += 1
        has_det, has_seg, lines = audit_label(lbl_path)

        if not (has_det and has_seg):
            continue  # clean file, skip

        stats["mixed"] += 1
        det_lines = [l for l in lines if len(l.split()) == 5]
        seg_lines = [l for l in lines if len(l.split()) > 5]
        stats["seg_rows_dropped"] += len(seg_lines)
        stats["det_rows_kept"] += len(det_lines)

        if not dry_run:
            lbl_path.write_text("\n".join(det_lines) + "\n" if det_lines else "", encoding="utf-8")
            stats["fixed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Strip seg rows from mixed YOLO label files.")
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["all", "train", "valid", "test"],
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report counts only, write nothing",
    )
    args = parser.parse_args()

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    grand = {"total": 0, "mixed": 0, "fixed": 0, "seg_rows_dropped": 0, "det_rows_kept": 0}
    for split in splits:
        split_dir = args.dataset_root / split
        if not split_dir.exists():
            print(f"[{split}] directory not found — skipping")
            continue
        stats = fix_split(split_dir, args.dry_run)
        for k in grand:
            grand[k] += stats[k]
        action = "WOULD fix" if args.dry_run else "Fixed"
        print(
            f"[{split}] {stats['total']} label files | "
            f"{stats['mixed']} mixed | "
            f"{action} {stats['fixed'] if not args.dry_run else stats['mixed']} | "
            f"seg rows dropped: {stats['seg_rows_dropped']} | "
            f"det rows kept: {stats['det_rows_kept']}"
        )

    print(
        f"\nTOTAL: {grand['total']} label files | "
        f"{grand['mixed']} mixed | "
        f"{'DRY RUN' if args.dry_run else 'Fixed: ' + str(grand['fixed'])} | "
        f"seg rows dropped: {grand['seg_rows_dropped']} | "
        f"det rows kept: {grand['det_rows_kept']}"
    )
    if args.dry_run:
        print("\nRun without --dry-run to apply the fix.")


if __name__ == "__main__":
    main()
