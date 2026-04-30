"""
make_slides.py
Generate docs/demo_slides.pptx — 8-slide Silent Failure Detection demo deck.
Run from project root: python docs/make_slides.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Colour palette ─────────────────────────────────────────────────────────
BG       = RGBColor(0x0A, 0x16, 0x28)   # dark navy
ACCENT   = RGBColor(0x00, 0xC9, 0xA7)   # teal
CARD     = RGBColor(0x15, 0x22, 0x38)   # card fill
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
YELLOW   = RGBColor(0xFF, 0xD7, 0x00)
RED_     = RGBColor(0xFF, 0x5C, 0x5C)
GREEN_   = RGBColor(0x4C, 0xD9, 0x7F)
GREY     = RGBColor(0xA0, 0xB0, 0xC0)
ORANGE   = RGBColor(0xFF, 0xA5, 0x00)

# Slide dimensions (16:9)
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

blank_layout = prs.slide_layouts[6]   # completely blank


# ── Helpers ────────────────────────────────────────────────────────────────

def add_bg(slide, color=BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def txbox(slide, text, left, top, width, height,
          font_size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT,
          bg_color=None, italic=False, wrap=True):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf  = box.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name  = "Calibri"
    if bg_color:
        box.fill.solid()
        box.fill.fore_color.rgb = bg_color
    return box


def rect(slide, left, top, width, height, fill_color=CARD, line_color=None, line_width=Pt(0)):
    from pptx.util import Pt as _Pt
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def hline(slide, left, top, width, color=ACCENT, thickness=Pt(1.5)):
    ln = slide.shapes.add_shape(1, left, top, width, thickness)
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    return ln


def badge(slide, text, left, top, width, height, bg_color=ACCENT, text_color=BG, font_size=11):
    r = rect(slide, left, top, width, height, fill_color=bg_color)
    txbox(slide, text, left, top, width, height,
          font_size=font_size, bold=True, color=text_color,
          align=PP_ALIGN.CENTER, bg_color=None)
    return r


def slide_header(slide, title, subtitle=None):
    """Standard slide header: accent line + title."""
    hline(slide, Inches(0.5), Inches(0.55), Inches(12.33))
    txbox(slide, title,
          Inches(0.5), Inches(0.65), Inches(12.33), Inches(0.7),
          font_size=28, bold=True, color=WHITE)
    if subtitle:
        txbox(slide, subtitle,
              Inches(0.5), Inches(1.35), Inches(12.33), Inches(0.4),
              font_size=14, color=ACCENT)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 1  — Title
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)

# Top accent bar
hline(s, Inches(0), Inches(0), Inches(13.33), color=ACCENT, thickness=Pt(4))

# Main title
txbox(s, "Silent Failure Detection", Inches(0.6), Inches(1.0), Inches(12), Inches(1.1),
      font_size=48, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
txbox(s, "for Clinical Machine Learning",
      Inches(0.6), Inches(2.0), Inches(12), Inches(0.7),
      font_size=32, bold=False, color=ACCENT, align=PP_ALIGN.LEFT)
txbox(s, "Early-warning uncertainty monitoring across 4 UQ methods, 3 failure modes, 2 clinical datasets",
      Inches(0.6), Inches(2.75), Inches(11), Inches(0.5),
      font_size=14, color=GREY, align=PP_ALIGN.LEFT)

# Three stat boxes
box_w, box_h = Inches(2.6), Inches(1.6)
stats = [
    ("4", "UQ Methods"),
    ("3", "Failure Modes"),
    ("2", "Clinical Datasets"),
]
for i, (num, lbl) in enumerate(stats):
    bx = Inches(0.6) + i * Inches(3.0)
    by = Inches(4.8)
    rect(s, bx, by, box_w, box_h, fill_color=CARD, line_color=ACCENT, line_width=Pt(1.5))
    txbox(s, num, bx, by + Inches(0.15), box_w, Inches(0.85),
          font_size=40, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    txbox(s, lbl, bx, by + Inches(0.95), box_w, Inches(0.5),
          font_size=13, color=WHITE, align=PP_ALIGN.CENTER)

# Bottom line
hline(s, Inches(0), Inches(7.3), Inches(13.33), color=CARD, thickness=Pt(3))
txbox(s, "Clinical AI Safety  ·  Uncertainty Quantification  ·  Distribution Shift Detection",
      Inches(0.6), Inches(7.1), Inches(12), Inches(0.35),
      font_size=10, color=GREY, align=PP_ALIGN.LEFT)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 2  — The Problem
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "The Problem: Models Fail Silently in Clinical Deployment")

# Problem statement
txbox(s, "A deployed model scores 87% accuracy in validation. Six months later, the data pipeline "
         "shifts, labels drift, or sensors degrade — but no alert fires. Accuracy quietly erodes to 72%. "
         "Clinicians trust predictions that are no longer reliable.",
      Inches(0.5), Inches(1.8), Inches(12.3), Inches(0.9),
      font_size=14, color=GREY)

# Timeline boxes
timeline_items = [
    ("Month 0", "Model deployed\nACC = 87%", GREEN_),
    ("Month 2", "Data pipeline\ndrift begins", YELLOW),
    ("Month 4", "Accuracy drops\nto ~72%", ORANGE),
    ("Month 6", "Clinician notices\nmisdiagnoses", RED_),
]
for i, (t, lbl, col) in enumerate(timeline_items):
    bx = Inches(0.5) + i * Inches(3.1)
    rect(s, bx, Inches(3.0), Inches(2.8), Inches(1.7), fill_color=CARD, line_color=col, line_width=Pt(2))
    txbox(s, t, bx, Inches(3.05), Inches(2.8), Inches(0.4),
          font_size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
    txbox(s, lbl, bx, Inches(3.45), Inches(2.8), Inches(0.9),
          font_size=13, color=WHITE, align=PP_ALIGN.CENTER)
    if i < 3:
        txbox(s, "▶", Inches(0.5) + i * Inches(3.1) + Inches(2.85),
              Inches(3.55), Inches(0.3), Inches(0.4),
              font_size=18, color=GREY, align=PP_ALIGN.CENTER)

# Key question
rect(s, Inches(0.5), Inches(5.1), Inches(12.3), Inches(0.9), fill_color=RGBColor(0x1A, 0x30, 0x50),
     line_color=ACCENT, line_width=Pt(1.5))
txbox(s, "Key Question: Can uncertainty signals warn us BEFORE accuracy drops — giving time to intervene?",
      Inches(0.65), Inches(5.15), Inches(12.0), Inches(0.8),
      font_size=15, bold=True, color=ACCENT, align=PP_ALIGN.LEFT)

# Three failure modes
fm_items = [
    ("Covariate Shift", "Feature distribution drifts\n(e.g. sensor calibration change)"),
    ("Label Noise", "Target labels corrupted\n(e.g. transcription errors)"),
    ("Feature Missingness", "Input columns zeroed out\n(e.g. pipeline data loss)"),
]
for i, (title, desc) in enumerate(fm_items):
    bx = Inches(0.5) + i * Inches(4.2)
    rect(s, bx, Inches(6.1), Inches(3.9), Inches(1.15), fill_color=CARD)
    txbox(s, title, bx + Inches(0.1), Inches(6.12), Inches(3.7), Inches(0.4),
          font_size=12, bold=True, color=ACCENT)
    txbox(s, desc, bx + Inches(0.1), Inches(6.5), Inches(3.7), Inches(0.6),
          font_size=10, color=GREY)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 3  — Our Approach
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "Our Approach: Four Uncertainty Quantification Methods",
             subtitle="Each method produces a per-sample uncertainty score fed into a KS-test alarm")

methods = [
    ("Conformal\n(MAPIE)",
     "Split conformal\nprediction sets.\n1 − max softmax\nconfidence.",
     "Assumption-free\ncoverage guarantee"),
    ("NGBoost\nEntropy",
     "Gradient-boosted\nprobabilistic model.\nBernoulli entropy\nH(p).",
     "Native distributional\noutput"),
    ("MC Dropout\nMLP",
     "100 stochastic\nforward passes.\nVariance of\npredictions.",
     "Bayesian approx.\nvia dropout"),
    ("TabTransformer",
     "Attention-based\ntabular encoder.\nEntropy of mean\nprediction†.",
     "†Residual+LayerNorm\nsuppresses MC variance"),
]
card_w = Inches(2.9)
for i, (title, body, note) in enumerate(methods):
    cx = Inches(0.5) + i * Inches(3.18)
    rect(s, cx, Inches(1.85), card_w, Inches(4.8), fill_color=CARD,
         line_color=ACCENT, line_width=Pt(1.5))
    # method number circle
    txbox(s, str(i+1), cx + Inches(0.1), Inches(1.9), Inches(0.45), Inches(0.45),
          font_size=16, bold=True, color=BG, align=PP_ALIGN.CENTER,
          bg_color=ACCENT)
    txbox(s, title, cx + Inches(0.1), Inches(2.4), card_w - Inches(0.2), Inches(0.85),
          font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txbox(s, body, cx + Inches(0.15), Inches(3.3), card_w - Inches(0.3), Inches(1.7),
          font_size=11, color=GREY, align=PP_ALIGN.CENTER)
    hline(s, cx + Inches(0.2), Inches(5.05), card_w - Inches(0.4), color=ACCENT, thickness=Pt(0.5))
    txbox(s, note, cx + Inches(0.1), Inches(5.15), card_w - Inches(0.2), Inches(0.8),
          font_size=9, color=ACCENT, italic=True, align=PP_ALIGN.CENTER)

# Alarm mechanism box
rect(s, Inches(0.5), Inches(6.75), Inches(12.3), Inches(0.55), fill_color=RGBColor(0x1A, 0x30, 0x50))
txbox(s, "Alarm: 3 consecutive KS-test rejections (p < 0.05) comparing corrupted vs. clean uncertainty distributions",
      Inches(0.65), Inches(6.77), Inches(12.0), Inches(0.5),
      font_size=12, color=ACCENT, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 4  — The Experiment
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "The Experiment: 2 Datasets × 3 Failure Modes × 10 Severity Levels",
             subtitle="α ∈ {0.0, 0.1, … 0.9}  ·  severity 0.0 = clean data, 0.9 = maximum corruption")

panels = [
    ("Covariate Shift",
     "Exponential reweighting\non feature 0.\nShifts marginal distribution\nof one input variable.",
     "DETECTABLE", GREEN_, "KS test on uncertainty\nscores detects shift\nat α ≥ 0.7 (Pima)"),
    ("Label Noise",
     "Random α-fraction\nof labels flipped.\nTarget corrupted but\nfeatures unchanged.",
     "UNDETECTABLE", RED_,  "Feature-derived uncertainty\ncannot see label-only\ncorruption"),
    ("Feature Missingness",
     "Column 0 zeroed after\nStandardScaler.\nEquivalent to imputing\nthe training mean.",
     "UNDETECTABLE", ORANGE, "Zero-fill = training-mean\nimputation → no feature\ndistribution shift"),
]
card_w = Inches(3.9)
for i, (title, desc, badge_text, badge_col, note) in enumerate(panels):
    cx = Inches(0.45) + i * Inches(4.3)
    rect(s, cx, Inches(1.85), card_w, Inches(5.1), fill_color=CARD,
         line_color=badge_col, line_width=Pt(1.5))
    txbox(s, title, cx + Inches(0.1), Inches(1.95), card_w - Inches(0.2), Inches(0.55),
          font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txbox(s, desc, cx + Inches(0.1), Inches(2.6), card_w - Inches(0.2), Inches(1.5),
          font_size=11, color=GREY, align=PP_ALIGN.CENTER)
    # badge
    badge_w = Inches(1.8)
    badge_left = cx + (card_w - badge_w) / 2
    rect(s, badge_left, Inches(4.25), badge_w, Inches(0.38), fill_color=badge_col)
    txbox(s, badge_text, badge_left, Inches(4.27), badge_w, Inches(0.34),
          font_size=10, bold=True, color=BG, align=PP_ALIGN.CENTER)
    txbox(s, note, cx + Inches(0.1), Inches(4.75), card_w - Inches(0.2), Inches(0.9),
          font_size=9, color=GREY, italic=True, align=PP_ALIGN.CENTER)

txbox(s, "Cleveland Heart Disease (n≈90 test set): KS test underpowered for covariate shift — documented limitation",
      Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.35),
      font_size=9, color=GREY, italic=True, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 5  — Key Results
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "Key Results: Detection Delay on Covariate Shift (Pima Dataset)")

# Stat callouts
stats5 = [
    ("70.1%", "Pima Baseline ACC"),
    ("79.0%", "Pima Baseline AUC"),
    ("86.7%", "Cleveland Baseline ACC"),
    ("89.4%", "Cleveland Baseline AUC"),
]
sw = Inches(2.9)
for i, (val, lbl) in enumerate(stats5):
    sx = Inches(0.5) + i * Inches(3.1)
    rect(s, sx, Inches(1.85), sw, Inches(1.0), fill_color=CARD,
         line_color=ACCENT, line_width=Pt(1))
    txbox(s, val, sx, Inches(1.9), sw, Inches(0.55),
          font_size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    txbox(s, lbl, sx, Inches(2.45), sw, Inches(0.35),
          font_size=10, color=GREY, align=PP_ALIGN.CENTER)

# Detection delay table header
txbox(s, "Detection Delay Table (δ = α_alarm − α_drop)  ·  Pima, Covariate Shift",
      Inches(0.5), Inches(3.1), Inches(12.3), Inches(0.4),
      font_size=13, bold=True, color=WHITE)
txbox(s, "Positive delay = alarm fires AFTER accuracy drops (late warning)",
      Inches(0.5), Inches(3.45), Inches(12.3), Inches(0.3),
      font_size=10, color=ORANGE, italic=True)

# Table
headers = ["Method", "α Alarm", "α Drop", "Delay δ", "Interpretation"]
rows5 = [
    ("Conformal (MAPIE)",  "0.8", "0.5", "+0.3", "Late — alarm fires 3 steps after drop"),
    ("NGBoost Entropy",    "0.7", "0.5", "+0.2", "Late — alarm fires 2 steps after drop"),
    ("MC Dropout MLP",     "0.8", "0.5", "+0.3", "Late — alarm fires 3 steps after drop"),
    ("TabTransformer",     "0.7", "0.5", "+0.2", "Late — alarm fires 2 steps after drop"),
]
col_widths = [Inches(2.4), Inches(1.1), Inches(1.1), Inches(1.0), Inches(5.5)]
col_lefts  = [Inches(0.5) + sum(col_widths[:i]) for i in range(5)]
row_h      = Inches(0.45)

# Header row
for j, (hdr, cw, cl) in enumerate(zip(headers, col_widths, col_lefts)):
    rect(s, cl, Inches(3.85), cw, row_h, fill_color=RGBColor(0x1A, 0x30, 0x50))
    txbox(s, hdr, cl + Inches(0.05), Inches(3.88), cw - Inches(0.1), row_h - Inches(0.05),
          font_size=11, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

for r_i, row in enumerate(rows5):
    ry = Inches(3.85) + (r_i + 1) * row_h
    bg_c = CARD if r_i % 2 == 0 else RGBColor(0x18, 0x28, 0x40)
    for j, (cell, cw, cl) in enumerate(zip(row, col_widths, col_lefts)):
        rect(s, cl, ry, cw, row_h, fill_color=bg_c)
        fc = ORANGE if j == 3 else WHITE
        txbox(s, cell, cl + Inches(0.05), ry + Inches(0.05),
              cw - Inches(0.1), row_h - Inches(0.05),
              font_size=10, color=fc, align=PP_ALIGN.CENTER if j != 0 else PP_ALIGN.LEFT)

# Take-away
rect(s, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.65),
     fill_color=RGBColor(0x1A, 0x30, 0x50), line_color=ORANGE, line_width=Pt(1.5))
txbox(s, "⚠  All four methods detect covariate shift — but only AFTER the accuracy has already dropped. "
         "Label noise & feature missingness remain undetectable by uncertainty alone.",
      Inches(0.65), Inches(6.63), Inches(12.0), Inches(0.59),
      font_size=11, color=ORANGE, align=PP_ALIGN.LEFT)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 6  — Interface Demo
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "Streamlit Interface: Real-Time Uncertainty Monitor",
             subtitle="http://localhost:8501  ·  Pre-computed results + live model inference")

# Sidebar mockup
rect(s, Inches(0.5), Inches(1.85), Inches(2.5), Inches(5.0), fill_color=RGBColor(0x0D, 0x1B, 0x30),
     line_color=GREY, line_width=Pt(0.5))
txbox(s, "⚙ Sidebar Controls", Inches(0.6), Inches(1.9), Inches(2.3), Inches(0.4),
      font_size=11, bold=True, color=ACCENT)
sidebar_items = [
    "📊 Dataset",
    "   pima / cleveland",
    "💥 Failure Mode",
    "   covariate_shift …",
    "🎚 Alarm Threshold",
    "   uncertainty slider",
    "📋 Methods to display",
    "   multiselect (4)",
    "🔢 MC / TT passes",
    "   1–200 slider",
    "▶ Rerun Evaluation",
]
for k, item in enumerate(sidebar_items):
    fc = ACCENT if item.startswith("▶") else (WHITE if not item.startswith("   ") else GREY)
    txbox(s, item, Inches(0.6), Inches(2.35) + k * Inches(0.32), Inches(2.3), Inches(0.3),
          font_size=9, color=fc)

# Panel 1
rect(s, Inches(3.2), Inches(1.85), Inches(4.7), Inches(2.35), fill_color=CARD)
txbox(s, "Panel 1 — Uncertainty Monitor", Inches(3.3), Inches(1.9), Inches(4.5), Inches(0.35),
      font_size=10, bold=True, color=ACCENT)
txbox(s, "Plotly line chart\nUncertainty vs. Severity α\n──── alarm threshold (hline)\n─·─ accuracy drop (vline)\n▒▒▒ detection window (vrect)",
      Inches(3.3), Inches(2.3), Inches(4.5), Inches(1.6),
      font_size=9, color=GREY)

# Panel 2
rect(s, Inches(8.1), Inches(1.85), Inches(4.7), Inches(2.35), fill_color=CARD)
txbox(s, "Panel 2 — Detection Summary Table", Inches(8.2), Inches(1.9), Inches(4.5), Inches(0.35),
      font_size=10, bold=True, color=ACCENT)
txbox(s, "Colour-coded delay table\n🟢 ≤ 0 (early warning)\n🟡 ≤ 0.2 (slight late)\n🔴 > 0.2 or None (missed)",
      Inches(8.2), Inches(2.3), Inches(4.5), Inches(1.6),
      font_size=9, color=GREY)

# Panel 3
rect(s, Inches(3.2), Inches(4.35), Inches(9.6), Inches(1.5), fill_color=CARD)
txbox(s, "Panel 3 — Model Status  (live inference at α = 0.5)",
      Inches(3.3), Inches(4.4), Inches(9.4), Inches(0.35),
      font_size=10, bold=True, color=ACCENT)
statuses = [
    ("Conformal\n🚨 ALARM", RED_),
    ("NGBoost\n🚨 ALARM", RED_),
    ("MC Dropout\n🚨 ALARM", RED_),
    ("TabTransformer\n🚨 ALARM", RED_),
]
for k, (lbl, col) in enumerate(statuses):
    bx = Inches(3.3) + k * Inches(2.4)
    rect(s, bx, Inches(4.8), Inches(2.2), Inches(0.75), fill_color=RGBColor(0x3A, 0x0A, 0x0A),
         line_color=col, line_width=Pt(1))
    txbox(s, lbl, bx + Inches(0.05), Inches(4.82), Inches(2.1), Inches(0.7),
          font_size=9, color=col, align=PP_ALIGN.CENTER)

txbox(s, "Status at covariate_shift α=0.5: all four methods correctly fire 🚨 ALARM",
      Inches(3.2), Inches(5.65), Inches(9.6), Inches(0.3),
      font_size=9, color=GREY, italic=True, align=PP_ALIGN.CENTER)

txbox(s, "📸  See docs/interface_screenshot.png for the live captured interface",
      Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.3),
      font_size=9, color=GREY, italic=True, align=PP_ALIGN.LEFT)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 7  — Responsible AI
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "Responsible AI Reflection",
             subtitle="Safety, fairness, and limitations in clinical deployment")

grid = [
    ("⚠  Known Limitations",
     ORANGE,
     "• Label noise: KS test blind to label-only corruption\n"
     "• Feature missingness = mean imputation → no shift\n"
     "• Cleveland n≈90: KS test underpowered\n"
     "• All delays positive: alarms fire AFTER accuracy drops"),
    ("🔒  Safety Recommendations",
     ACCENT,
     "• Supplement with calibration-error monitoring\n"
     "• Use sentinel imputation (−3σ) for missingness\n"
     "• Lower KS consecutive threshold from 3→2 for small n\n"
     "• Deploy human-in-the-loop review at first warning"),
    ("⚖  Fairness Considerations",
     GREEN_,
     "• Covariate shift may affect demographic subgroups\n  differently — monitor per-subgroup uncertainty\n"
     "• Threshold choice (p<0.05) trades recall vs. precision\n"
     "• Training data bias propagates through all methods"),
    ("📋  Deployment Checklist",
     WHITE,
     "• Calibrate alarm thresholds on hold-out period\n"
     "• Log all predictions + uncertainty scores\n"
     "• Retrain protocol triggered by 3 consecutive alarms\n"
     "• Regulatory documentation for each failure mode"),
]
cell_w = Inches(5.9)
cell_h = Inches(2.3)
for i, (title, col, body) in enumerate(grid):
    cx = Inches(0.5) + (i % 2) * Inches(6.4)
    cy = Inches(1.85) + (i // 2) * Inches(2.5)
    rect(s, cx, cy, cell_w, cell_h, fill_color=CARD, line_color=col, line_width=Pt(1.5))
    txbox(s, title, cx + Inches(0.15), cy + Inches(0.1), cell_w - Inches(0.3), Inches(0.4),
          font_size=13, bold=True, color=col)
    txbox(s, body, cx + Inches(0.15), cy + Inches(0.5), cell_w - Inches(0.3), cell_h - Inches(0.6),
          font_size=10, color=GREY)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 8  — Conclusion & Next Steps
# ══════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_bg(s)
slide_header(s, "Conclusion & Next Steps")

# Summary bullets
txbox(s, "What We Built", Inches(0.5), Inches(1.85), Inches(5.8), Inches(0.4),
      font_size=14, bold=True, color=ACCENT)
summary_points = [
    "End-to-end silent failure detection pipeline for clinical ML",
    "4 UQ methods: Conformal, NGBoost, MC Dropout MLP, TabTransformer",
    "Controlled injection of 3 clinical failure modes at 10 severity levels",
    "KS-test alarm with detection-delay metric (δ = α_alarm − α_drop)",
    "Streamlit dashboard with live inference + pre-computed sweep results",
    "Full reproducibility: seeded, config-driven, pinned requirements",
]
for k, pt in enumerate(summary_points):
    txbox(s, "▸  " + pt,
          Inches(0.5), Inches(2.35) + k * Inches(0.42), Inches(5.8), Inches(0.38),
          font_size=11, color=WHITE if k % 2 == 0 else GREY)

# Next steps
txbox(s, "Next Steps", Inches(7.0), Inches(1.85), Inches(5.8), Inches(0.4),
      font_size=14, bold=True, color=ACCENT)
next_items = [
    ("1", "Sentinel imputation", "Replace zero-fill with −3σ to make\nfeature missingness detectable"),
    ("2", "Calibration monitor", "ECE tracking to catch label-noise\ncorruption missed by KS test"),
    ("3", "Subgroup fairness", "Per-demographic uncertainty profiles\nfor equitable alarm thresholds"),
    ("4", "Prospective trial", "Deploy on real clinical stream with\nground-truth label lag validation"),
]
for k, (num, title, desc) in enumerate(next_items):
    cy = Inches(2.35) + k * Inches(1.15)
    rect(s, Inches(7.0), cy, Inches(5.8), Inches(1.0), fill_color=CARD)
    txbox(s, num, Inches(7.05), cy + Inches(0.15), Inches(0.4), Inches(0.4),
          font_size=16, bold=True, color=BG, align=PP_ALIGN.CENTER,
          bg_color=ACCENT)
    txbox(s, title, Inches(7.55), cy + Inches(0.05), Inches(5.1), Inches(0.38),
          font_size=12, bold=True, color=WHITE)
    txbox(s, desc, Inches(7.55), cy + Inches(0.43), Inches(5.1), Inches(0.5),
          font_size=10, color=GREY)

# Bottom accent
hline(s, Inches(0), Inches(7.3), Inches(13.33), color=ACCENT, thickness=Pt(3))
txbox(s, "Silent Failure Detection for Clinical ML  ·  github.com/arkha/silent_failure_project",
      Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3),
      font_size=9, color=GREY, align=PP_ALIGN.CENTER)


# ── Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "demo_slides.pptx")
prs.save(out_path)
print(f"Saved {prs.slides.__len__()} slides -> {out_path}")
