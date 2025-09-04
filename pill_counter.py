import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# ---------------- Page ----------------
st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# ---------------- Helpers ----------------
def downscale(img_bgr, max_dim=1600):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def illumination_correction(gray):
    # normalize uneven lighting
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    blur = np.clip(blur, 1, 255)
    norm = (gray.astype(np.float32) / blur.astype(np.float32)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def global_contrast_ok(gray):
    # RMS contrast (std dev), and Otsu separation check
    std = gray.std()
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fraction of pixels near the Otsu threshold (ambiguous mid-tones)
    t = int(_)
    band = cv2.inRange(gray, max(0, t-10), min(255, t+10))
    mid_frac = band.mean() / 255.0
    # Heuristics tuned for phone photos on benches/trays
    return (std >= 22) and (mid_frac <= 0.36), dict(std=std, mid_frac=mid_frac, thr=t)

def segment_mask(gray, block_size=23, c_val=10):
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, c_val
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def split_touching(mask):
    # watershed split
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_bg = cv2.dilate(mask, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    sep = np.zeros_like(mask)
    sep[markers > 1] = 255
    return sep

def filter_pill_contours(contours, img_area):
    # dynamic size limits
    min_area = int(img_area * 0.006)
    max_area = int(img_area * 0.12)
    keep = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        # solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        if area / hull_area < 0.82:
            continue
        # aspect ratio (reject long edges/strings)
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar > 3.0:
            continue
        keep.append(c)
    return keep

def quality_score(contours):
    # higher is better: many similar-sized, solid contours
    if not contours:
        return 0.0
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    mean, std = areas.mean(), areas.std()
    if mean <= 1:
        return 0.0
    # coefficient of variation (lower is better), invert to score
    cv = std / mean
    cv_term = 1.0 / (1.0 + cv)  # 0..1
    # more contours increases confidence but with diminishing returns
    n_term = 1.0 - math.exp(-len(contours)/12.0)
    return 0.6 * cv_term + 0.4 * n_term  # 0..1

def run_pass(bgr, block_size, c_val):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illumination_correction(gray)
    mask = segment_mask(gray, block_size=block_size, c_val=c_val)
    mask = split_touching(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = mask.shape[0] * mask.shape[1]
    pills = filter_pill_contours(contours, img_area)
    score = quality_score(pills)
    return pills, score

def draw_output(bgr, contours, count):
    out = bgr.copy()
    cv2.drawContours(out, contours, -1, (0,0,255), 3)
    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------------- Pipeline with auto-retake + contrast mask ----------------
if file_to_use is not None:
    pil = Image.open(file_to_use).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1600)

    # Contrast sanity check
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grayN = illumination_correction(gray0)
    ok, metrics = global_contrast_ok(grayN)

    if not ok:
        st.warning(
            "Low contrast detected. Move away from glare, use a plainer background, "
            "or add more even light. Then retake the photo."
        )
        # Still try a count, but mark as low confidence

    # Auto-retake logic: try 3 parameter sets and pick the best
    passes = [
        (23, 10),  # default
        (27, 8),   # slightly larger neighborhoods
        (21, 12),  # slightly stronger separation
    ]
    results = []
    for bs, c in passes:
        pills, score = run_pass(bgr, bs, c)
        results.append((pills, score, bs, c))

    # Choose the pass with highest quality score
    pills_best, score_best, bs_best, c_best = max(results, key=lambda r: r[1])
    count_best = len(pills_best)

    # Sanity: if the best two passes disagree by >20%, warn user
    counts = sorted([len(r[0]) for r in results])
    if counts[-1] == 0:
        agree_ok = False
    else:
        agree_ok = (counts[-1] - counts[0]) / max(1, counts[-1]) <= 0.20

    rgb_out = draw_output(bgr, pills_best, count_best)
    st.image(rgb_out, caption="Detected pills with count overlay", use_container_width=True)
    st.success(f"Pill count: {count_best}")

    if not ok:
        st.caption(f"Note: low-contrast photo (std={metrics['std']:.1f}, mid-tone frac={metrics['mid_frac']:.2f}).")
    if not agree_ok:
        st.info("Counts varied across checks. If this seems off, try 1) plain background, 2) less glare, 3) separate tight clusters.")
