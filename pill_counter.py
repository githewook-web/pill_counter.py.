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
def downscale(img_bgr, max_dim=1800):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def illumination_correction(gray):
    # Normalize uneven lighting; helps fluorescent glare on stainless
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    blur = np.clip(blur, 1, 255)
    norm = (gray.astype(np.float32) / blur.astype(np.float32)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def global_contrast_ok(gray):
    # Simple photo quality gate
    std = float(gray.std())
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    band = cv2.inRange(gray, max(0, int(t)-10), min(255, int(t)+10))
    mid_frac = float(band.mean() / 255.0)
    return (std >= 20.0) and (mid_frac <= 0.40), dict(std=std, mid_frac=mid_frac, thr=int(t))

def segment_adaptive(gray, inv=True, block_size=23, c_val=10):
    # inv=True counts light pills on dark counters; inv=False for darker pills on light trays
    thresh_type = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type,
                                block_size, c_val)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def split_touching(mask, fg_frac=0.45):
    # Watershed separation for light-to-moderately touching pills
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_frac * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_bg = cv2.dilate(mask, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    sep = np.zeros_like(mask)
    sep[markers > 1] = 255
    return sep

def filter_pill_contours(contours, img_area, tight=False):
    # Size gates adapt to image size. Tight mode for small batches to protect accuracy.
    if tight:
        min_area_frac, max_area_frac = 0.004, 0.10
        min_solidity, max_ar = 0.84, 2.6
    else:
        min_area_frac, max_area_frac = 0.006, 0.12
        min_solidity, max_ar = 0.82, 3.0

    min_area = int(img_area * min_area_frac)
    max_area = int(img_area * max_area_frac)

    keep = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        # Solidity rejects jagged edges, tray rims, and wood grain
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        if (area / hull_area) < min_solidity:
            continue

        # Aspect ratio allows round/oval/capsule, rejects long edges/strings
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar > max_ar:
            continue

        keep.append(c)
    return keep

def quality_score(contours):
    # Confidence heuristic: consistent sizes and enough items
    if not contours:
        return 0.0
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    mean, std = areas.mean(), areas.std()
    if mean <= 1:
        return 0.0
    cv = std / mean
    cv_term = 1.0 / (1.0 + cv)           # 0..1 (lower variability is better)
    n_term  = 1.0 - math.exp(-len(contours)/12.0)  # 0..1 with diminishing returns
    return 0.6 * cv_term + 0.4 * n_term

def run_pass(bgr, inv, block_size, c_val, fg_frac, tight=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illumination_correction(gray)
    mask = segment_adaptive(gray, inv=inv, block_size=block_size, c_val=c_val)
    mask = split_touching(mask, fg_frac=fg_frac)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = mask.shape[0] * mask.shape[1]
    pills = filter_pill_contours(contours, img_area, tight=tight)
    score = quality_score(pills)
    return pills, score

def draw_output(bgr, contours, count):
    out = bgr.copy()
    cv2.drawContours(out, contours, -1, (0,0,255), 3)  # red outline
    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------------- Pipeline ----------------
if file_to_use is not None:
    pil = Image.open(file_to_use).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1800)

    # Contrast gate
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grayN = illumination_correction(gray0)
    contrast_ok, metrics = global_contrast_ok(grayN)
    if not contrast_ok:
        st.warning("Low contrast detected. If the count seems off, try a plainer background or reduce glare and retake.")

    # Pass sets:
    # inv=True targets light pills on dark counters; inv=False catches darker pills on light trays.
    # Small-batch path runs tighter filters and stronger separation.
    # Large-batch path is a bit looser to avoid over-splitting.
    passes_primary = [
        # default balanced
        (True, 23, 10, 0.45, False),
        # darker pills on light tray
        (False, 23, 10, 0.45, False),
        # larger neighborhood smooths noise on stainless
        (True, 27, 8, 0.45, False),
    ]

    # Run initial passes
    results = []
    for inv, bs, c, fg, tight in passes_primary:
        pills, score = run_pass(bgr, inv, bs, c, fg, tight=tight)
        results.append((pills, score, inv, bs, c, fg, tight))

    # Choose best by quality, get preliminary count
    pills_best, score_best, inv_best, bs_best, c_best, fg_best, tight_best = max(results, key=lambda r: r[1])
    prelim_count = len(pills_best)

    # Decide mode from preliminary count
    small_batch = prelim_count <= 50

    # Add targeted passes depending on batch size
    if small_batch:
        passes_secondary = [
            (True, 21, 12, 0.40, True),   # tighter min area, stronger split
            (True, 25, 10, 0.42, True),
            (False, 21, 12, 0.40, True),
        ]
    else:
        passes_secondary = [
            (True, 29, 8, 0.48, False),   # avoid over-splitting large sets
            (False, 25, 10, 0.48, False),
        ]

    for inv, bs, c, fg, tight in passes_secondary:
        pills, score = run_pass(bgr, inv, bs, c, fg, tight=tight)
        results.append((pills, score, inv, bs, c, fg, tight))

    # Final pick
    pills_final, score_final, inv_f, bs_f, c_f, fg_f, tight_f = max(results, key=lambda r: r[1])
    counts = sorted([len(r[0]) for r in results])
    final_count = len(pills_final)

    # Disagreement warning
    spread = (counts[-1] - counts[0]) / max(1, counts[-1]) if counts[-1] else 1.0
    if small_batch:
        agree_ok = spread <= 0.10   # stricter for small counts
    else:
        agree_ok = spread <= 0.20

    # Render
    rgb_out = draw_output(bgr, pills_final, final_count)
    st.image(rgb_out, caption="Detected pills with count overlay", use_container_width=True)
    st.success(f"Pill count: {final_count}")

    # Helpful notes without clutter
    if not contrast_ok:
        st.caption(f"Note: low-contrast photo (std={metrics['std']:.1f}, mid-tone frac={metrics['mid_frac']:.2f}).")
    if not agree_ok:
        st.info("Counts varied across checks. If this seems off, try a plainer background, reduce glare, or separate tight clusters.")
