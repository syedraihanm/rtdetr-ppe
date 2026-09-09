"""
verify_detect.py — Test /detect and /ask endpoints with a real image.
"""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Pick a real test image
imgs = list(Path("test/images").glob("*.jpg")) or list(Path("test/images").glob("*.png"))
assert imgs, "No test images found in test/images/!"
img_path = imgs[0]
print(f"Using image: {img_path.name}\n")

with open(img_path, "rb") as f:
    img_bytes = f.read()

KNOWN_CLASSES = {
    "Eye_protection", "Foot_protection", "Hand_protection", "Head_protection",
    "No_eye_protection", "No_foot_protection", "No_hand_protection",
    "No_head_protection", "No_respiratory_protection", "No_safety_vest",
    "Respiratory_protection", "Safety_vest", "Person"
}

# ─── /detect ────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST: POST /detect")
print("=" * 60)
r = client.post("/detect", files={"file": (img_path.name, img_bytes, "image/jpeg")})
assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
d = r.json()
assert "detections" in d
assert "image_width" in d
assert "image_height" in d
print(f"  [OK] Status: 200")
print(f"  [OK] {len(d['detections'])} detection(s) | {d['image_width']}x{d['image_height']} px")
print()

errors = []
for det in d["detections"][:8]:
    cls = det.get("class") or det.get("class_")
    conf = det["confidence"]
    box = det["box"]
    if len(box) != 4:
        errors.append(f"Box should have 4 values: {box}")
        continue
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):
        errors.append(f"Bad box coords (x2 > x1 and y2 > y1 required): {box}")
    print(f"    {cls:<28} conf={conf:.4f}  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

if errors:
    for e in errors:
        print(f"  [FAIL] {e}")
    sys.exit(1)
print()
print("  [OK] Bounding box format verified (xyxy pixel coords, x2>x1, y2>y1)")

# ─── /ask (no helmet) ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST: POST /ask — 'Is anyone not wearing a helmet?'")
print("=" * 60)
r = client.post(
    "/ask",
    files={"file": (img_path.name, img_bytes, "image/jpeg")},
    data={"question": "Is anyone not wearing a helmet?"},
)
assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
a = r.json()
assert "answer" in a
assert "used_detection" in a
assert "confidence" in a
assert "supporting_detections" in a
print(f"  [OK] Status: 200")
print(f"  [OK] answer     : {a['answer']}")
print(f"  [OK] confidence : {a['confidence']}")
print(f"  [OK] used_detection: {a['used_detection']}")
print(f"  [OK] supporting_detections: {len(a['supporting_detections'])} items")

# ─── /ask (count) ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST: POST /ask — 'How many people are on site?'")
print("=" * 60)
r = client.post(
    "/ask",
    files={"file": (img_path.name, img_bytes, "image/jpeg")},
    data={"question": "How many people are on site?"},
)
assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
a = r.json()
print(f"  [OK] Status: 200")
print(f"  [OK] answer     : {a['answer']}")
print(f"  [OK] confidence : {a['confidence']}")
print(f"  [OK] used_detection: {a['used_detection']}")

# ─── /ask (most common) ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST: POST /ask — 'What PPE is most common?'")
print("=" * 60)
r = client.post(
    "/ask",
    files={"file": (img_path.name, img_bytes, "image/jpeg")},
    data={"question": "What PPE is most common?"},
)
assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
a = r.json()
print(f"  [OK] Status: 200")
print(f"  [OK] answer     : {a['answer']}")
print(f"  [OK] confidence : {a['confidence']}")

# ─── /ask (off-topic → guardrail) ────────────────────────────────────────────
print()
print("=" * 60)
print("TEST: POST /ask — Off-topic question (should not need detection)")
print("=" * 60)
r = client.post(
    "/ask",
    files={"file": (img_path.name, img_bytes, "image/jpeg")},
    data={"question": "What is the capital of France?"},
)
assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
a = r.json()
assert a["used_detection"] == False, f"Off-topic question should have used_detection=False, got {a}"
print(f"  [OK] Status: 200")
print(f"  [OK] answer          : {a['answer']}")
print(f"  [OK] used_detection  : {a['used_detection']}  (correctly did NOT run detection)")

print()
print("=" * 60)
print("ALL ENDPOINT TESTS PASSED")
print("=" * 60)
