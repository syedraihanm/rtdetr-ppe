# Project Brief: PPE Detection API + Reasoning Layer (RT-DETR)

> Use this document as your build spec. Feed each section to Antigravity as you reach that stage — don't dump the whole thing at once, or the model will produce shallow, generic code across all parts instead of depth in any one part.

---

## 0. Constraints Recap (do not violate)
- No LangChain/LangGraph/CrewAI/AutoGen/agent frameworks — Part B logic must be hand-written Python + raw LLM API calls only.
- No AutoML/no-code training.
- Must fine-tune **RT-DETR** specifically (Ultralytics RT-DETR recommended — cleanest API).
- At least one class must be **not** in COCO's 80 classes.
- Training must be fully reproducible (exact steps, env, hardware, time, hyperparams) or Part A is capped at 50%.
- 5 failure cases with root-cause analysis are graded — don't hide weaknesses, document them.

---

## 1. Domain & Dataset Choice

**Domain:** Construction-site PPE (Personal Protective Equipment) compliance detection.

**Classes (non-COCO ones in bold):**
- `Person` (COCO class — keep for context/counting)
- **`Hardhat`**
- **`NO-Hardhat`**
- **`Safety-Vest`**
- **`NO-Safety-Vest`**
- **`Mask`** / **`NO-Mask`** (optional, drop if you want a smaller label set)

**Recommended dataset:** "Construction Site Safety Image Dataset" on Roboflow Universe (search that exact name — there are a few near-duplicates; pick the one with ~10 classes, ~3,000+ images, already in YOLO format with a valid/test split). Roboflow lets you export directly in COCO JSON or YOLO txt — grab **COCO format** since Ultralytics RT-DETR and pycocotools both consume it cleanly.

Why this dataset over scraping your own:
- You have 5 days total, most of which needs to go to training/eval/API/reasoning-layer/memo — not labeling.
- It already has NO-Hardhat / NO-Safety-Vest as explicit negative classes, which is exactly the kind of thing that makes Part B's "is anyone missing PPE" question answerable from structured detections.
- It's real-world, non-COCO, and has known quality issues (label noise, class imbalance, some blur) — which gives you **honest** material for the 5-failure-case memo requirement instead of having to manufacture failures.

**Fallback/second option** if that dataset looks thin when you inspect it: "Hard Hat Workers Dataset" (Kaggle, ~5k images, Hardhat/Head/Person) — narrower but cleaner, good backup.

**Mid-project pivot note for your memo:** if you start with one and switch, that's explicitly fine per the brief — just write one sentence on why.

---

## 2. Train/Val/Test Split Strategy

- Use the dataset's existing split if it's sane (check class balance per split first — don't trust it blindly).
- If not, do an 80/10/10 stratified-by-class split (stratify approximately, since object detection can't perfectly stratify multi-label images) — script this yourself, don't hand-split in a GUI, so it's reproducible.
- Hold out the test set completely from training/hyperparameter tuning — only touch it once, at the end, for the self-reported metrics in your memo. The hidden eval set is separate from all of this.
- Justification to include in memo: class imbalance (NO-Hardhat is usually rarer than Hardhat — note the actual counts), and why you didn't rebalance via oversampling (or did, and why).

---

## 3. Training Setup (given your hardware)

Your i5-1335U / 16GB RAM / MX550 (2GB VRAM) **cannot reasonably fine-tune RT-DETR** — 2GB VRAM will OOM even at batch size 1-2 with RT-DETR's transformer decoder memory footprint. Do NOT try to force this locally past a smoke test.

**Plan:**
1. **Local machine:** environment setup, data inspection/EDA, writing training script, smoke-testing the script on ~10 images for 1 epoch on CPU (just to catch bugs before burning cloud GPU hours) and building the FastAPI app + Part B logic.
2. **Kaggle Notebooks** (free, 30 GPU-hrs/week, T4 x2, no credit card needed) or **Google Colab free tier** (T4, ~12hr sessions) for actual training.
3. Model: `rtdetr-l.pt` (Ultralytics) as the pretrained checkpoint to fine-tune from — good accuracy/speed tradeoff; drop to `rtdetr-r18` config if you want faster epochs and can accept a small mAP hit (worth trying both, report the tradeoff — this is good failure-analysis material).

**Suggested hyperparameters (starting point, tune from here):**
- Image size: 640
- Batch size: 8–16 (T4 16GB can handle this for rtdetr-l)
- Epochs: 60–100 with early stopping on val mAP
- Optimizer: AdamW, lr ~1e-4 with cosine decay (Ultralytics defaults are reasonable — don't over-tune, document what you changed and why)
- Freeze backbone for first N epochs, unfreeze for fine-tuning — note whether this helped or not
- Augmentation: mosaic, horizontal flip, slight color jitter — be cautious with vertical flip/rotation for this domain (a rotated hardhat может look wrong) — worth a one-line note in memo either way

**Log for reproducibility (put in a `TRAINING.md` in the repo):**
- Exact Ultralytics/PyTorch/CUDA versions
- GPU type and hours used
- Full hyperparameter dict (dump the actual `args.yaml` Ultralytics saves)
- Wall-clock training time
- Random seed

---

## 4. Evaluation (Part A)

- Use Ultralytics' built-in `.val()` for mAP50, mAP50-95, per-class precision/recall — this uses COCO-style eval under the hood.
- Also manually inspect a confusion matrix (Ultralytics generates one) specifically for Hardhat vs NO-Hardhat and Vest vs NO-Vest — these are your hardest pairs (visually similar, differ by a small cue), and confusion here is exactly the kind of "what metrics don't tell you" discussion the memo wants.
- Report metrics honestly even if mediocre — the brief explicitly rewards 78% mAP + real failure analysis over 95% mAP + no acknowledged weaknesses.

**Failure case categories to actively hunt for (need 5 for memo):**
1. Small/distant objects (person far from camera, hardhat is a few pixels)
2. Occlusion (hand/tool covering the vest logo area)
3. Class confusion (a light-colored hardhat vs. bare head in bright light; orange vest vs. orange safety cone)
4. Blur/motion
5. Lighting extremes (backlit workers, night/indoor low light if present in data)

Go find one real example image + prediction for each, screenshot it with the box drawn, and write 2-3 sentences of root cause per case. Don't write these generically before you've actually looked at your model's outputs.

---

## 5. FastAPI App — Endpoint 1: `/detect`

**Request:** multipart image upload
**Response (JSON):**
```json
{
  "detections": [
    {"class": "Hardhat", "confidence": 0.91, "box": [x1, y1, x2, y2]},
    {"class": "NO-Safety-Vest", "confidence": 0.77, "box": [x1, y1, x2, y2]},
    {"class": "Person", "confidence": 0.95, "box": [x1, y1, x2, y2]}
  ],
  "image_width": 1280,
  "image_height": 720
}
```
Keep box format documented explicitly (xyxy in pixel coords) — reviewers will run this against a hidden set, so ambiguity here costs you.

---

## 6. FastAPI App — Endpoint 2: `/ask` (Part B reasoning layer)

**Request:**
```json
{"question": "Is anyone not wearing a helmet?"}
```
(plus the image, same as `/detect`)

**Response:**
```json
{
  "answer": "Yes, 1 of 4 people detected is not wearing a hardhat.",
  "used_detection": true,
  "confidence": "high",
  "supporting_detections": [ ... ]
}
```

### Hand-written decision layer — no framework, three explicit stages:

**Stage 1 — Intent Router** (`route_intent(question: str) -> dict`)
Write this as a single direct call to an LLM API (raw `requests`/`httpx` call to Anthropic or OpenAI's `/messages` or `/chat/completions` endpoint — NOT an SDK-wrapped agent, just a plain HTTP call with a tightly constrained system prompt). Prompt it to return **strict JSON only**:
```json
{"needs_detection": true, "query_type": "presence_check", "target_class": "Hardhat", "negated": true}
```
`query_type` should cover at minimum: `count`, `presence_check`, `most_common`, `general_no_detection_needed`. This is the "structured reasoning" the brief wants — you're not asking the LLM to freeform-answer, you're asking it to classify + extract slots, then Python does the actual logic deterministically in Stage 2. This split is exactly what "hand-written decision layer" is testing.

**Stage 2 — Structured Reasoning** (`reason_over_detections(detections: list, intent: dict) -> dict`)
Pure Python, no LLM call needed here for the well-defined query types:
- `count`: filter detections by target_class, len()
- `presence_check` + `negated=True`: check if any `NO-{target_class}` detections exist
- `most_common`: Counter over detected classes
- Compute an aggregate confidence for the answer (e.g., min or mean confidence of the detections actually used to answer)

Only fall back to a second LLM call for genuinely open-ended phrasing you can't map to the four query types — and even then, feed it the structured detection list, not the raw image, so it's reasoning over your model's output, not re-doing vision itself.

**Stage 3 — Confidence Guardrail** (`apply_guardrail(answer, confidence, threshold=0.5) -> dict`)
If the detections needed to answer have confidence below threshold, or the target class has zero detections in a question that presupposes it's answerable (and the image plausibly shows the scene), return:
```json
{"answer": "I can't confidently answer this from the detections — confidence too low.", "used_detection": true, "confidence": "low"}
```
**This exact behavior — one concrete example where it correctly says "insufficient information" — is a required memo item.** Pick a real low-confidence case from your test set, don't fabricate one.

---

## 7. Repo Structure

```
├── README.md                 # setup + run instructions, sample curl requests
├── TRAINING.md                # reproducibility log (hardware, time, hyperparams, seed)
├── requirements.txt
├── data/
│   └── prepare_split.py       # your train/val/test split script
├── train.py                   # RT-DETR fine-tuning script
├── evaluate.py                 # mAP / P/R / confusion matrix script
├── app/
│   ├── main.py                 # FastAPI app, both endpoints
│   ├── detection.py             # loads model, runs inference
│   └── reasoning.py              # Stage 1/2/3 from section 6
├── weights/                    # or a download script if too large for git
├── memo.pdf                     # max 2 pages, see section 8
└── Dockerfile                    # bonus points
```

---

## 8. Memo Outline (max 2 pages — write this last, after you have real results)

1. Domain/dataset choice + sourcing/labeling (2-3 sentences)
2. Split strategy + justification (2-3 sentences)
3. Metrics table + 2-3 sentences on what they do/don't tell you (e.g., "high overall mAP is driven by the Person class; Hardhat/NO-Hardhat mAP is X points lower, which matters more for the actual safety-compliance use case than the aggregate number")
4. **5 failure cases** — each with a thumbnail/description + root cause, 2-3 sentences each
5. Part B logic summary + the one required "insufficient information" example, described concretely (what was asked, what the detector returned, why it was insufficient, what was returned)
6. (If applicable) mid-project pivot note

---

## 9. Suggested 5-Day Timeline

- **Day 1:** Dataset acquisition, inspection, split script, EDA (class counts, image sizes, a few visual sanity checks). Write training script, smoke-test locally on CPU.
- **Day 2:** Move to Kaggle/Colab, run first real training run, watch for OOM/config issues.
- **Day 3:** Finish training (possibly a second run with adjusted hyperparams), run evaluation, dig for failure cases.
- **Day 4:** Build FastAPI `/detect`, then `/ask` with the 3-stage reasoning layer. Test both endpoints locally.
- **Day 5:** Dockerize (bonus), write TRAINING.md + README, write memo, final repo cleanup, double-check reproducibility instructions actually work from scratch.

---

## 10. What to hand Antigravity, one prompt at a time

Don't paste this whole doc as one prompt. Suggested prompt sequence:
1. "Set up a Python project for fine-tuning Ultralytics RT-DETR on a Roboflow COCO-format dataset at [path]. Write `train.py` using these hyperparameters: [paste section 3]. Include full logging of config/time/seed to TRAINING.md format."
2. "Write `evaluate.py` using Ultralytics `.val()` plus a confusion matrix specifically for these class pairs: Hardhat/NO-Hardhat, Safety-Vest/NO-Safety-Vest."
3. "Build a FastAPI app with the `/detect` endpoint per this exact request/response spec: [paste section 5]."
4. "Implement the 3-stage reasoning layer in `app/reasoning.py` per this spec, using a raw HTTP call to [Anthropic/OpenAI] — no SDK agent wrappers, no LangChain: [paste section 6]."

Feeding it in stages like this — with you reviewing each output before moving on — is also exactly what you'll need to be able to defend line-by-line in the verbal round.
