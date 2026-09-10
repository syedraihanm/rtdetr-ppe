"""
demo_live.py — Comprehensive End-to-End System Demonstration
Simulates real client requests against the FastAPI application loading weights/best.pt.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=" * 70)
print("1. SYSTEM HEALTH CHECK (GET /health)")
print("=" * 70)
resp = client.get("/health")
print(f"HTTP Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))

# Test Image 1
img_path1 = Path("test/images/2008_008526_jpg.rf.aa97dfd8d6642f41fb1859b06bc6849b.jpg")
print("\n" + "=" * 70)
print(f"2. OBJECT DETECTION (POST /detect) on {img_path1.name}")
print("=" * 70)
with open(img_path1, "rb") as f:
    resp = client.post("/detect?conf=0.30", files={"file": (img_path1.name, f, "image/jpeg")})
data = resp.json()
print(f"HTTP Status: {resp.status_code}")
print(f"Image resolution: {data['image_width']}x{data['image_height']}")
print(f"Total detections: {len(data['detections'])}")
print("\nTop detected PPE objects and workers:")
for d in data['detections'][:8]:
    print(f"  • {d['class']:<25} conf={d['confidence']:.3f} box={[round(x, 1) for x in d['box']]}")

print("\n" + "=" * 70)
print("3. SAFETY REASONING ENGINE (POST /ask)")
print("=" * 70)

queries = [
    ("Is anyone not wearing a helmet?", img_path1),
    ("How many people are on site?", img_path1),
    ("What is the capital of Japan?", img_path1),
]

for q, img_p in queries:
    with open(img_p, "rb") as f:
        resp = client.post(
            "/ask",
            data={"question": q},
            files={"file": (img_p.name, f, "image/jpeg")},
        )
    print(f"\nQuestion: \"{q}\"")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")

# Test Image 2: Required Real Insufficient Information Case
img_path2 = Path("test/images/-1680-_png_jpg.rf.73cee3e264b17ce5579750df4e4610f4.jpg")
print("\n" + "=" * 70)
print("4. REQUIRED 'INSUFFICIENT INFORMATION' GUARDRAIL CASE")
print(f"   Image: {img_path2.name}")
print("=" * 70)
q_edge = "Is anyone wearing a safety vest?"
with open(img_path2, "rb") as f:
    resp = client.post(
        "/ask",
        data={"question": q_edge},
        files={"file": (img_path2.name, f, "image/jpeg")},
    )
print(f"Question: \"{q_edge}\"")
print(f"Response: {json.dumps(resp.json(), indent=2)}")

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
