"""
generate_memo_pdf.py — Generate the 2-page Technical Memo PDF per Section 8 of Project Brief.
Uses ReportLab with strict 2-page budget, professional styling, metrics tables, and failure analysis.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def build_pdf(filename="memo.pdf"):
    # Margins: 0.45 in (32 pt) all around to guarantee clean 2-page fit
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )
    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.0,
        leading=9,
        textColor=colors.HexColor("#0f172a"),
    )
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#ffffff"),
        alignment=1,
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1e293b"),
        alignment=0,
    )
    td_center = ParagraphStyle(
        "TDCenter",
        parent=td_style,
        alignment=1,
    )

    story = []

    # ── HEADER ───────────────────────────────────────────────────────────────
    story.append(Paragraph("RT-DETR-Based Object Detection for Safety Equipment", title_style))
    story.append(Paragraph("<b>Author:</b> Syed Mohamed Raihan &nbsp;|&nbsp; <b>Model:</b> Ultralytics RT-DETR-L &nbsp;|&nbsp; <b>Repo:</b> github.com/syedraihanm/rtdetr-ppe &nbsp;|&nbsp; <b>Date:</b> September 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=0, spaceAfter=5))

    # ── SECTION 1: Domain & Dataset Choice ────────────────────────────────────
    story.append(Paragraph("1. Domain/Dataset Choice, Sourcing & Labeling", h1_style))
    story.append(Paragraph(
        "Selected the Roboflow Universe <i>Construction Site Safety v2</i> dataset (CC BY 4.0), with 10,875 high-resolution "
        "construction scene images annotated across 12 safety equipment classes. Because the original dataset labeled equipment "
        "without annotating workers. Expanded the dataset to 13 classes by automatically labeling <b>Person</b> bounding boxes using "
        "a COCO-pretrained YOLO11 detector with a confidence threshold of &ge; 0.50. This yields 11,018 worker instances "
        "across 62.4% of images, enabling the model to reason over both equipment presence and worker context in a single unified detector.",
        body_style
    ))

    # ── SECTION 2: Split Strategy ─────────────────────────────────────────────
    story.append(Paragraph("2. Split Strategy & Justification", h1_style))
    story.append(Paragraph(
        "The dataset follows an <b>82.4% train (8,956 images) / 11.8% valid (1,279 images) / 5.8% test (640 images)</b> partition. "
        "Splits are strictly disjoint at the image level. Construction video frame sequences were kept contiguous within individual splits "
        "to prevent temporal leakage between training and evaluation. Stratification checks verified that rare classes (e.g. Foot protection, "
        "Respiratory protection) maintain consistent relative frequency across splits.",
        body_style
    ))

    # ── SECTION 3: Metrics Table & Analysis ──────────────────────────────────
    story.append(Paragraph("3. Quantitative Evaluation Metrics & Operational Meaning", h1_style))
    story.append(Paragraph(
        "I stopped training at <b>Epoch 71</b> (after 9h 40m on a Kaggle Tesla T4 GPU) as validation metrics showed clear saturation after peak fitness at <b>Epoch 65</b>. Final evaluation on validation and test splits:",
        body_style
    ))

    metrics_data = [
        [Paragraph("Split / Metric", th_style), Paragraph("mAP@50", th_style), Paragraph("mAP@50-95", th_style), Paragraph("Precision", th_style), Paragraph("Recall", th_style), Paragraph("Key Class Drivers", th_style)],
        [Paragraph("<b>Validation (1,279 imgs)</b>", td_style), Paragraph("<b>79.90%</b>", td_center), Paragraph("<b>52.35%</b>", td_center), Paragraph("78.40%", td_center), Paragraph("79.49%", td_center), Paragraph("Peak fitness at Epoch 65; stopped at Epoch 71 (saturation)", td_style)],
        [Paragraph("<b>Test Set (640 imgs)</b>", td_style), Paragraph("<b>79.19%</b>", td_center), Paragraph("<b>50.99%</b>", td_center), Paragraph("78.05%", td_center), Paragraph("78.80%", td_center), Paragraph("Unseen test split generalization", td_style)],
        [Paragraph("• Head Protection (Hardhat)", td_style), Paragraph("80.27%", td_center), Paragraph("51.47%", td_center), Paragraph("86.19%", td_center), Paragraph("84.08%", td_center), Paragraph("Direct compliance driver; high precision", td_style)],
        [Paragraph("• No Head Protection (Bare)", td_style), Paragraph("82.82%", td_center), Paragraph("49.56%", td_center), Paragraph("81.21%", td_center), Paragraph("81.82%", td_center), Paragraph("Direct violation driver; catches 81.8% of unhelmeted heads", td_style)],
        [Paragraph("• Safety Vest / No Vest", td_style), Paragraph("74.92% / 63.01%", td_center), Paragraph("49.98% / 39.51%", td_center), Paragraph("76.67% / 77.99%", td_center), Paragraph("77.09% / 62.06%", td_center), Paragraph("High-contrast workwear causes vest confusion", td_style)],
        [Paragraph("• Foot Protection (Boots)", td_style), Paragraph("94.96%", td_center), Paragraph("74.13%", td_center), Paragraph("80.87%", td_center), Paragraph("93.88%", td_center), Paragraph("Strong ground-contrast features", td_style)],
        [Paragraph("• Person Class (Auto-labeled)", td_style), Paragraph("58.23%", td_center), Paragraph("48.62%", td_center), Paragraph("56.90%", td_center), Paragraph("61.03%", td_center), Paragraph("Occluded / truncated bodies limit person score", td_style)],
    ]
    t_metrics = Table(metrics_data, colWidths=[1.8*inch, 0.85*inch, 0.95*inch, 0.85*inch, 0.8*inch, 2.25*inch])
    t_metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("ALIGN", (1, 1), (4, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Operational Interpretation:</b> Aggregate mAP@50 (79.2%) is heavily bolstered by distinct items like boots (95.0%) and eye protection (87.9%). "
        "However, for site safety auditability, the critical operational metric is the paired confusion between positive compliance and negative violation classes. "
        "Confusion matrix analysis reveals <b>41 instances of False Compliance on Head Protection</b> (bare head classified as hardhat) and <b>41 instances on Safety Vest</b>. "
        "In life-critical environments, False Compliance is dramatically more hazardous than False Alarm, directly establishing the requirement for Stage 3 confidence filtering.",
        body_style
    ))

    # ── SECTION 4: Mid-Project Pivot Note ─────────────────────────────────────
    story.append(Paragraph("4. Mid-Project Data Engineering Pivot", h1_style))
    story.append(Paragraph(
        "During baseline inspection, inspection detected that 4,990 annotation files (45.9% of the dataset) contained corrupted rows combining "
        "5-parameter bounding box rows with raw polygon segmentation coordinates. Ultralytics silently dropped these images during training. "
        "Developing <font name='Courier'>fix_mixed_labels.py</font> cleansed corrupted lines, recovering 21,529 valid bounding boxes across all splits. "
        "Additionally, discovering that Roboflow lacked a Person class prompted developing an auto-labeling pipeline (<font name='Courier'>add_person_class.py</font>), "
        "preventing the need for a brittle two-stage cascaded detector at runtime.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Embedded Figure on Page 1: Detection Sample + Normalized Confusion Matrix
    story.append(RLImage("docs/images/page1_detection_preview.jpg", width=7.4*inch, height=2.05*inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<b>Figure 1:</b> <i>Left:</i> Real-time RT-DETR-L detections demonstrating simultaneous Person anchor context and PPE bounding box predictions. "
        "<i>Right:</i> Normalized validation confusion matrix illustrating high on-diagonal accuracy and off-diagonal compliance confusions.",
        callout_style
    ))

    # ── PAGE BREAK ────────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── SECTION 5: Failure Cases Analysis ─────────────────────────────────────
    story.append(Paragraph("5. Systematic Failure Analysis (5 Real Test Cases)", h1_style))
    story.append(Paragraph(
        "Empirical inspection of the 640 held-out test predictions revealed five distinct structural failure modes:",
        body_style
    ))

    failures_data = [
        [
            Paragraph("Visual Evidence", th_style),
            Paragraph("Case / Test Image", th_style),
            Paragraph("Ground Truth vs Predicted", th_style),
            Paragraph("Root Cause & Safety Consequence", th_style),
        ],
        [
            RLImage("docs/images/failure_cases/thumb_case1.jpg", width=1.75*inch, height=0.86*inch),
            Paragraph("<b>Case 1: Shadow False Compliance</b><br/><font name='Courier' size=5.8>2008_008526...jpg</font>", td_style),
            Paragraph("GT: <b>No_head_protection</b><br/>Pred: <b>Head_protection</b> (conf 0.30)", td_style),
            Paragraph("Worker in shadow wearing dark beanie; brim curvature mimics hardhat dome. <i>Consequence: Dangerous false compliance clears unhelmeted worker.</i>", td_style),
        ],
        [
            RLImage("docs/images/failure_cases/thumb_case2.jpg", width=1.75*inch, height=0.86*inch),
            Paragraph("<b>Case 2: High-Vis Workwear Confusion</b><br/><font name='Courier' size=5.8>001425...7b40.jpg</font>", td_style),
            Paragraph("GT: <b>No_safety_vest</b><br/>Pred: <b>Safety_vest</b> (conf 0.86)", td_style),
            Paragraph("High-vis work shirt with tool straps mimics retroreflective vest tape. <i>Consequence: Safety vest violation missed by auditor.</i>", td_style),
        ],
        [
            RLImage("docs/images/failure_cases/thumb_case3.jpg", width=1.75*inch, height=0.86*inch),
            Paragraph("<b>Case 3: Extreme Scale Disparity</b><br/><font name='Courier' size=5.8>construction-3...148.jpg</font>", td_style),
            Paragraph("GT: <b>Hand_protection</b> & <b>Eye</b><br/>Pred: <i>Zero detections</i> for PPE", td_style),
            Paragraph("High crane shot with workers &lt; 80 px; gloves &lt; 14 px collapse below stride-32 feature pyramid. <i>Consequence: False alarms on distant PPE.</i>", td_style),
        ],
        [
            RLImage("docs/images/failure_cases/thumb_case4.jpg", width=1.75*inch, height=0.86*inch),
            Paragraph("<b>Case 4: Truncated Edge Silhouette</b><br/><font name='Courier' size=5.8>-1680...73ce.jpg</font>", td_style),
            Paragraph("GT: <b>Person</b> + <b>Safety_vest</b><br/>Pred: Vest (conf 0.39), Person missed", td_style),
            Paragraph("Worker stepping into frame on left edge (65% cropped). Model misses worker silhouette and vest falls below 0.50 threshold.", td_style),
        ],
        [
            RLImage("docs/images/failure_cases/thumb_case5.jpg", width=1.75*inch, height=0.86*inch),
            Paragraph("<b>Case 5: Equipment False Positive</b><br/><font name='Courier' size=5.8>4c43875b...29c6.jpg</font>", td_style),
            Paragraph("GT: <b>Background (Mixer)</b><br/>Pred: <b>Head_protection</b> (conf 0.36)", td_style),
            Paragraph("Yellow convex hydraulic cap on mixer misclassified as helmet due to specular highlight and hemispherical geometry.", td_style),
        ],
    ]
    t_failures = Table(failures_data, colWidths=[1.85*inch, 1.35*inch, 1.6*inch, 2.7*inch])
    t_failures.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_failures)
    story.append(Spacer(1, 4))

    # ── SECTION 6: Part B Reasoning Logic & Required Concrete Example ─────────
    story.append(Paragraph("6. Hand-Written 3-Stage Decision Layer & Insufficient Information Guardrail", h1_style))
    story.append(Paragraph(
        "To comply with architectural constraints forbidding agentic frameworks (no LangChain/CrewAI), <font name='Courier'>app/reasoning.py</font> "
        "executes a hand-written three-stage deterministic pipeline: "
        "<b>Stage 1 (Intent Routing)</b> extracts query intent (<font name='Courier'>presence_check</font>, <font name='Courier'>count</font>, <font name='Courier'>most_common</font>, or <font name='Courier'>off_topic</font>) via raw HTTP to LLM with rule-based regex fallback. "
        "<b>Stage 2 (Structured Reasoning)</b> evaluates bounding boxes deterministically, pairing positive and negative compliance classes. "
        "<b>Stage 3 (Confidence Guardrail)</b> prevents dangerous false declarations by enforcing conservative confidence thresholds.",
        body_style
    ))

    # Concrete required example box
    story.append(Paragraph("<b>Required Memo Case Study: Real 'Insufficient Information' Production Execution</b>", body_bold))
    guardrail_box_data = [
        [
            Paragraph(
                "<b>Input Test Image:</b> <font name='Courier'>-1680-_png_jpg.rf.73cee3e264b17ce5579750df4e4610f4.jpg</font><br/>"
                "<b>User Query:</b> <i>\"Is anyone wearing a safety vest?\"</i><br/>"
                "<b>Detector Output:</b> 1 detection &rarr; <font name='Courier'>Safety_vest</font> at <font name='Courier'>[0.0, 40.9, 22.8, 172.6]</font> with <b>conf = 0.3942</b>.<br/>"
                "<b>Why Evidence Was Insufficient:</b> The worker is heavily edge-cropped on the border. The detection confidence (0.394) is strictly below the safety threshold (0.50), while person confidence is absent.<br/>"
                "<b>API Response Returned:</b><br/>"
                "<font name='Courier' size=6.8>{\"answer\": \"I can't confidently answer this from the detections — confidence too low.\", \"used_detection\": true, \"confidence\": \"low\"}</font>",
                callout_style
            )
        ]
    ]
    t_box = Table(guardrail_box_data, colWidths=[7.5*inch])
    t_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284c7")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 4))

    # ── SECTION 7: Deployment Recommendations ────────────────────────────────
    story.append(Paragraph("7. Deployment Architecture & Operational Guardrails", h1_style))
    story.append(Paragraph(
        "<b>1. Asymmetric Thresholding:</b> Deploy with split thresholds: set <i>conf = 0.65</i> for declaring compliance (<font name='Courier'>Head_protection</font>) "
        "and <i>conf = 0.35</i> for flagging potential violations (<font name='Courier'>No_head_protection</font>). This guarantees a conservative bias towards safety inspections. "
        "<b>2. Multi-Frame Temporal Consensus:</b> In video streams, require PPE violations to persist for &ge; 5 consecutive frames before alerting safety managers, mitigating edge-crop glitches. "
        "<b>3. Hardware Efficiency:</b> RT-DETR-L executes in 774 ms on CPU (laptop test) and 12.8 ms on Tesla T4, comfortably meeting 30 FPS real-time site camera monitoring requirements.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated 2-page Technical Memo: {filename}")


if __name__ == "__main__":
    build_pdf()
