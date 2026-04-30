"""
make_video.py
Creates docs/demo_video.mp4 — a 3-minute silent slideshow demo.

Sections:
  0:00–0:12  Title slide (rendered)
  0:12–0:35  Problem slide (rendered)
  0:35–1:05  Approach slide (rendered)
  1:05–1:25  Experiment slide (rendered)
  1:25–1:55  Key Results slide (rendered)
  1:55–2:55  Live app screenshots (4 states)
  2:55–3:10  Conclusion slide (rendered)

Run from project root:
    python docs/make_video.py
Requires: matplotlib, Pillow, opencv-python-headless, playwright
"""

import os, time, textwrap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

# ── Config ──────────────────────────────────────────────────────────────────
OUT_PATH   = os.path.join(os.path.dirname(__file__), "demo_video.mp4")
SHOT_DIR   = os.path.join(os.path.dirname(__file__), "_shots")
W, H       = 1920, 1080
FPS        = 24
APP_URL    = "http://localhost:8501"

os.makedirs(SHOT_DIR, exist_ok=True)

# ── Colour helpers ───────────────────────────────────────────────────────────
BG_COL    = (10,  22,  40)     # #0A1628  (PIL = R,G,B)
ACCENT    = (0,  201, 167)     # #00C9A7
CARD      = (21,  34,  56)     # #152238
WHITE     = (255, 255, 255)
GREY      = (140, 160, 180)
ORANGE    = (255, 165,  0)
RED_      = (255,  92,  92)
GREEN_    = (76,  217, 127)
YELLOW    = (255, 215,   0)


def pil_img(color=BG_COL):
    img = Image.new("RGB", (W, H), color)
    return img, ImageDraw.Draw(img)


def load_font(size, bold=False):
    """Try to load Calibri / Arial / fallback."""
    candidates = [
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arialbd.ttf"  if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text_wrapped(draw, text, x, y, max_w, font, fill=WHITE, align="left"):
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] > max_w and line:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=fill, align=align)
        bb = draw.textbbox((0, 0), ln, font=font)
        y += (bb[3] - bb[1]) + 6
    return y


def hline(draw, y, x0=60, x1=W-60, color=ACCENT, width=3):
    draw.rectangle([x0, y, x1, y + width], fill=color)


def card_rect(draw, x, y, w, h, fill=CARD, outline=None, outline_w=2):
    draw.rectangle([x, y, x+w, y+h], fill=fill)
    if outline:
        draw.rectangle([x, y, x+w, y+h], outline=outline, width=outline_w)


def caption_bar(draw, text, y=H-52, font_size=22):
    draw.rectangle([0, y, W, H], fill=(5, 12, 25))
    f = load_font(font_size)
    draw.text((W//2, y+14), text, font=f, fill=GREY, anchor="mm")


def img_to_frame(img):
    """PIL Image -> numpy BGR frame for cv2."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frames_from_img(img, seconds):
    frame = img_to_frame(img)
    return [frame] * int(FPS * seconds)


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE RENDERS
# ════════════════════════════════════════════════════════════════════════════

def slide_title():
    img, d = pil_img()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    f_big  = load_font(80, bold=True)
    f_sub  = load_font(46)
    f_sm   = load_font(26)
    f_stat = load_font(64, bold=True)
    f_lbl  = load_font(24)
    d.text((80, 120), "Silent Failure Detection", font=f_big, fill=WHITE)
    d.text((80, 220), "for Clinical Machine Learning", font=f_sub, fill=ACCENT)
    draw_text_wrapped(d,
        "Early-warning uncertainty monitoring across 4 UQ methods, 3 failure modes, 2 clinical datasets",
        80, 295, W-160, f_sm, fill=GREY)

    stats = [("4", "UQ Methods"), ("3", "Failure Modes"), ("2", "Clinical Datasets")]
    for i, (num, lbl) in enumerate(stats):
        bx = 80 + i * 420
        card_rect(d, bx, 600, 360, 220, fill=CARD, outline=ACCENT, outline_w=3)
        d.text((bx+180, 640), num, font=f_stat, fill=ACCENT, anchor="mm")
        d.text((bx+180, 730), lbl, font=f_lbl, fill=WHITE, anchor="mm")

    caption_bar(d, "Silent Failure Detection for Clinical ML  ·  Demo")
    return img


def slide_approach():
    img, d = pil_img()
    hline(d, 60)
    d.text((80, 75), "Our Approach: Four Uncertainty Quantification Methods", font=load_font(42, bold=True), fill=WHITE)

    methods = [
        ("1", "Conformal\n(MAPIE)", "Split conformal prediction.\n1 − max softmax confidence.\nAssumption-free coverage."),
        ("2", "NGBoost\nEntropy",   "Probabilistic gradient boosting.\nBernoulli entropy H(p).\nNative distributional output."),
        ("3", "MC Dropout\nMLP",    "100 stochastic forward passes.\nVariance of predictions.\nBayesian approximation."),
        ("4", "TabTransformer",     "Attention on tabular features.\nEntropy of mean prediction.\n†LayerNorm suppresses MC var."),
    ]
    cw, ch = 420, 460
    for i, (num, title, body) in enumerate(methods):
        cx = 60 + i * 465
        card_rect(d, cx, 160, cw, ch, fill=CARD, outline=ACCENT, outline_w=2)
        # circle number
        d.ellipse([cx+14, 174, cx+64, 224], fill=ACCENT)
        d.text((cx+39, 199), num, font=load_font(28, bold=True), fill=BG_COL, anchor="mm")
        y = 240
        for line in title.split("\n"):
            d.text((cx + cw//2, y), line, font=load_font(28, bold=True), fill=WHITE, anchor="mm")
            y += 38
        y += 10
        for line in body.split("\n"):
            d.text((cx + cw//2, y), line, font=load_font(21), fill=GREY, anchor="mm")
            y += 32

    d.rectangle([60, 665, W-60, 700], fill=(26, 48, 80))
    d.text((W//2, 682), "Alarm: 3 consecutive KS-test rejections (p < 0.05)", font=load_font(24), fill=ACCENT, anchor="mm")
    caption_bar(d, "Each method produces a per-sample uncertainty score fed into a KS-test alarm")
    return img


def slide_results():
    img, d = pil_img()
    hline(d, 60)
    d.text((80, 75), "Key Results: Detection Delay on Covariate Shift (Pima)", font=load_font(40, bold=True), fill=WHITE)

    # stat boxes
    stats = [
        ("70.1%", "Pima Baseline ACC"),
        ("79.0%", "Pima Baseline AUC"),
        ("86.7%", "Cleveland Baseline ACC"),
        ("89.4%", "Cleveland Baseline AUC"),
    ]
    sw = 420
    for i, (val, lbl) in enumerate(stats):
        sx = 60 + i * 465
        card_rect(d, sx, 165, sw, 120, fill=CARD, outline=ACCENT, outline_w=2)
        d.text((sx + sw//2, 200), val, font=load_font(44, bold=True), fill=ACCENT, anchor="mm")
        d.text((sx + sw//2, 248), lbl, font=load_font(20), fill=GREY, anchor="mm")

    # Table
    d.text((80, 320), "Detection Delay Table — Pima Dataset, Covariate Shift (δ = α_alarm − α_drop)", font=load_font(26, bold=True), fill=WHITE)
    d.text((80, 356), "Positive delay = alarm fires AFTER accuracy drop", font=load_font(20), fill=ORANGE)

    headers = ["Method", "α Alarm", "α Drop", "Delay δ"]
    rows = [
        ("Conformal (MAPIE)", "0.8", "0.5", "+0.3"),
        ("NGBoost Entropy",   "0.7", "0.5", "+0.2"),
        ("MC Dropout MLP",    "0.8", "0.5", "+0.3"),
        ("TabTransformer",    "0.7", "0.5", "+0.2"),
    ]
    col_w  = [540, 200, 200, 200]
    col_x  = [80]; [col_x.append(col_x[-1]+w) for w in col_w[:-1]]
    row_h  = 60
    hy     = 400
    for j, (hdr, cw, cx) in enumerate(zip(headers, col_w, col_x)):
        d.rectangle([cx, hy, cx+cw, hy+row_h], fill=(26, 48, 80))
        d.text((cx+cw//2, hy+row_h//2), hdr, font=load_font(22, bold=True), fill=ACCENT, anchor="mm")
    for ri, row in enumerate(rows):
        ry = hy + (ri+1) * row_h
        bg = CARD if ri % 2 == 0 else (24, 40, 64)
        for j, (cell, cw, cx) in enumerate(zip(row, col_w, col_x)):
            d.rectangle([cx, ry, cx+cw, ry+row_h], fill=bg)
            fc = ORANGE if j == 3 else WHITE
            d.text((cx+cw//2, ry+row_h//2), cell, font=load_font(22), fill=fc, anchor="mm")

    d.rectangle([60, 678, W-60, 730], fill=(26, 48, 80))
    draw_text_wrapped(d,
        "All four methods detect covariate shift — but only AFTER the accuracy has already dropped. "
        "Label noise & feature missingness remain undetectable by uncertainty alone.",
        80, 690, W-180, load_font(20), fill=ORANGE)
    caption_bar(d, "Pima Diabetes  ·  Covariate Shift  ·  KS-test alarm")
    return img


def slide_conclusion():
    img, d = pil_img()
    hline(d, 60)
    d.text((80, 75), "Conclusion & Next Steps", font=load_font(52, bold=True), fill=WHITE)

    points = [
        "End-to-end silent failure detection pipeline for clinical ML",
        "4 UQ methods: Conformal, NGBoost, MC Dropout MLP, TabTransformer",
        "KS-test alarm detects covariate shift — all methods fire late (+0.2 to +0.3)",
        "Label noise and feature missingness: not detectable by uncertainty alone",
        "Streamlit dashboard: live inference + pre-computed sweep + rerun button",
        "Fully reproducible: seeded (42), config-driven, pinned requirements",
    ]
    f = load_font(26)
    for i, pt in enumerate(points):
        d.text((100, 200 + i*72), "▸  " + pt, font=f, fill=WHITE if i%2==0 else GREY)

    next_items = [
        ("1", "Sentinel imputation",  "Use -3σ fill to make missingness detectable"),
        ("2", "Calibration monitor",  "ECE tracking catches label-noise corruption"),
        ("3", "Subgroup fairness",    "Per-demographic uncertainty profiles"),
        ("4", "Prospective trial",    "Real clinical stream validation"),
    ]
    for i, (num, title, desc) in enumerate(next_items):
        cx = 60 + (i % 2) * 940
        cy = 660 + (i // 2) * 130
        card_rect(d, cx, cy, 880, 112, fill=CARD)
        d.ellipse([cx+10, cy+12, cx+54, cy+56], fill=ACCENT)
        d.text((cx+32, cy+34), num, font=load_font(24, bold=True), fill=BG_COL, anchor="mm")
        d.text((cx+74, cy+20), title, font=load_font(24, bold=True), fill=WHITE)
        d.text((cx+74, cy+54), desc, font=load_font(20), fill=GREY)

    d.rectangle([0, H-6, W, H], fill=ACCENT)
    caption_bar(d, "Silent Failure Detection for Clinical ML")
    return img


# ════════════════════════════════════════════════════════════════════════════
#  BROWSER SCREENSHOTS
# ════════════════════════════════════════════════════════════════════════════

APP_STATES = [
    # (dataset, failure_mode, caption, wait_extra_ms)
    ("pima",      "covariate_shift", "Panel 1 — Uncertainty Monitor  |  pima / covariate_shift", 3000),
    ("pima",      "covariate_shift", "Panel 2 — Detection Summary Table  |  alarm fires late (+0.2 to +0.3)", 0),
    ("pima",      "label_noise",     "Panel 1 — label_noise  |  uncertainty lines flat — undetectable", 3000),
    ("pima",      "covariate_shift", "Panel 3 — Model Status  |  all four methods: ALARM at alpha=0.5", 0),
]

SCROLL_TARGETS = [
    None,           # Panel 1 — top of page
    0.35,           # Panel 2 — mid page
    None,           # Panel 1 again for label_noise
    0.70,           # Panel 3 — bottom
]


def take_app_screenshots():
    shots = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": W, "height": H})

        for idx, ((ds, fm, caption, extra_wait), scroll_frac) in enumerate(
            zip(APP_STATES, SCROLL_TARGETS)
        ):
            print(f"  App screenshot {idx+1}/{len(APP_STATES)}: {ds}/{fm}")
            page.goto(APP_URL)
            page.wait_for_load_state("networkidle", timeout=20000)
            page.wait_for_timeout(3500)

            # Select dataset
            try:
                sb = page.locator('[data-testid="stSelectbox"]').first
                sb.click(); page.wait_for_timeout(500)
                page.locator(f'li[role="option"]:has-text("{ds}")').first.click()
                page.wait_for_timeout(400)
            except Exception:
                pass

            # Select failure mode
            try:
                sbs = page.locator('[data-testid="stSelectbox"]').all()
                if len(sbs) > 1:
                    sbs[1].click(); page.wait_for_timeout(500)
                    page.locator(f'li[role="option"]:has-text("{fm}")').first.click()
                    page.wait_for_timeout(400)
            except Exception:
                pass

            if extra_wait:
                page.wait_for_timeout(extra_wait)

            if scroll_frac is not None:
                page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {scroll_frac})")
                page.wait_for_timeout(600)

            path = os.path.join(SHOT_DIR, f"app_{idx:02d}.png")
            page.screenshot(path=path, full_page=False)

            # Add caption bar
            img = Image.open(path).convert("RGB").resize((W, H))
            d = ImageDraw.Draw(img)
            d.rectangle([0, H-52, W, H], fill=(5, 12, 25))
            d.text((W//2, H-26), caption, font=load_font(22), fill=ACCENT, anchor="mm")
            img.save(path)
            shots.append(path)

        browser.close()
    return shots


# ════════════════════════════════════════════════════════════════════════════
#  VIDEO ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def cross_fade(frame_a, frame_b, n_frames=12):
    """Yield n_frames blending from frame_a to frame_b."""
    for i in range(n_frames):
        alpha = i / n_frames
        blended = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
        yield blended


def add_progress_bar(frame, progress, color=(0, 201, 167)):
    """Draw a thin progress bar at top of frame (mutates frame in-place)."""
    bar_w = int(W * progress)
    frame[0:4, 0:bar_w] = color
    return frame


def build_video(slide_imgs, app_shots, durations):
    """
    slide_imgs : list of PIL Images for rendered slides
    app_shots  : list of paths to app screenshots
    durations  : list of seconds per segment (len = len(slide_imgs) + len(app_shots))
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

    total_frames = sum(int(d * FPS) for d in durations)
    written = 0

    all_segments = []
    for i, img in enumerate(slide_imgs):
        all_segments.append(("pil", img))
    for path in app_shots:
        all_segments.append(("path", path))

    # Interleave slides and app shots per the script order:
    # title, approach, results, [4 app states], conclusion
    ordered = [
        ("pil", slide_imgs[0]),   # title
        ("pil", slide_imgs[1]),   # approach
        ("pil", slide_imgs[2]),   # results
        ("path", app_shots[0]),   # app: covariate_shift panel1
        ("path", app_shots[1]),   # app: covariate_shift panel2
        ("path", app_shots[2]),   # app: label_noise panel1
        ("path", app_shots[3]),   # app: covariate_shift panel3 (status)
        ("pil", slide_imgs[3]),   # conclusion
    ]
    # durations match
    seg_durations = durations  # passed in matching order

    prev_frame = None
    for (kind, src), dur in zip(ordered, seg_durations):
        if kind == "pil":
            frame = img_to_frame(src)
        else:
            pil = Image.open(src).convert("RGB").resize((W, H))
            frame = img_to_frame(pil)

        n = int(dur * FPS)

        # Cross-fade from previous segment
        if prev_frame is not None:
            fade_frames = list(cross_fade(prev_frame, frame, n_frames=10))
            for ff in fade_frames:
                prog = written / total_frames
                add_progress_bar(ff, prog)
                writer.write(ff)
                written += 1

        for fi in range(n):
            prog = written / total_frames
            f = frame.copy()
            add_progress_bar(f, prog)
            writer.write(f)
            written += 1

        prev_frame = frame

    writer.release()
    print(f"Written {written} frames ({written/FPS:.1f}s) -> {OUT_PATH}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[1/3] Rendering slide frames...")
    slides = [
        slide_title(),
        slide_approach(),
        slide_results(),
        slide_conclusion(),
    ]

    print("[2/3] Capturing app screenshots via Playwright...")
    app_shots = take_app_screenshots()

    # Durations per segment (seconds) — total ~185s (~3m05s)
    durations = [
        14,   # title
        22,   # approach
        22,   # results
        28,   # app: covariate_shift panel 1
        22,   # app: covariate_shift panel 2 (delay table)
        22,   # app: label_noise (flat)
        24,   # app: panel 3 status badges
        16,   # conclusion
    ]
    print(f"  Total planned duration: {sum(durations)}s ({sum(durations)/60:.1f} min)")

    print("[3/3] Assembling video...")
    build_video(slides, app_shots, durations)
    print(f"\nDone! Video saved to: {OUT_PATH}")
