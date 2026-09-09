"""verify_yaml.py — Validate data.yaml schema."""
import yaml
from pathlib import Path

with open("data.yaml") as f:
    d = yaml.safe_load(f)

assert d["nc"] == 13, f"nc should be 13, got {d['nc']}"
names = d["names"]
assert len(names) == 13, f"names should have 13 entries, got {len(names)}"
assert names[-1] == "Person", f"Last class should be Person, got {names[-1]}"
assert "Head_protection" in names
assert "No_head_protection" in names
assert "Safety_vest" in names
assert "No_safety_vest" in names

print("[OK] data.yaml: nc=13")
print(f"[OK] Classes: {names}")
print(f"[OK] Train path (relative): {d['train']}")
print(f"[OK] Val path (relative):   {d['val']}")
print(f"[OK] Test path (relative):  {d['test']}")

# Validate that Ultralytics dataset resolver finds the images
from ultralytics.data.utils import check_det_dataset
resolved = check_det_dataset("data.yaml")
for split in ("train", "val", "test"):
    split_path = Path(resolved[split])
    assert split_path.exists(), f"{split} path does not exist: {split_path}"
    n_images = len(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")))
    assert n_images > 0, f"No images found in {split_path}"
    print(f"[OK] {split} resolved: {split_path} ({n_images} images)")

print("\nAll data.yaml checks passed successfully!")
