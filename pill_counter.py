import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import math

# ---------- Page ----------
st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# ---------- Helpers ----------
def _safe_open_image(file):
    """Open safely, fix EXIF rotation, force RGB."""
    pil = Image.open(file)
    pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("RGB")
    return pil

def downscale(bgr, max_dim=1800):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _size_aware_sigma(gray):
    """Pick a blur sigma relative to size to avoid over/under-flattening."""
    h, w = gray.shape[:2]
    return max(7, min(45, round(min(h, w) / 24)))

def illum_norm(gray):
    """Flatten slow lighting gradients; robust for both small & large images."""
    sigma = _size_aware_sigma(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur = np.clip(blur, 1, 255)
    norm = (gray.astype(np.float32) / blur.astype(np.float32)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def tophat_enhance(gray):
    """Boost bright pill bodies vs soft shadows using white tophat."""
    h, w = gray.shape[:2]
    k = int(max(9, min(41, round(min(h, w) / 30))))  # odd-ish range
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    # Blend a little with normalized image to keep edges but suppress halos
    out = cv2.addWeighted(gray, 0.65, th, 0.35, 0)
    return out

def clear_border(mask, pct=0.02):
    """Zero a thin border so frames/edges never become one giant contour."""
    h, w = mask.shape[:2]
    m = int(round(min(h, w) * pct))
    if m < 1:
        return mask
    mask[:m, :] = 0
    mask[-m:, :] = 0
    mask[:, :m] = 0
    mask[:, -m:] = 0
    return mask

def floodfill_background(mask, fill_value=0):
    """
    Remove any foreground connected to the image border (tray/background)
    by flood-filling from the 4 corners on the *mask*'s foreground.
    """
    h, w = mask.shape[:2]
    m = mask.copy()
    ff = m.copy()
    # We flood on the *inverse* to carve away big connected background,
    # then invert back to the original polarity.
    inv = cv2.bitwise_not(ff)
    hpad, wpad = h + 2, w + 2
    ff_mask = np.zeros((hpad, wpad), dtype=np.uint8)

    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        ff_mask[...] = 0
        inv2 = inv.copy()
        cv2.floodFill(inv2, ff_mask, seedPoint=(seed[1], seed[0]), newVal=0)
        inv = inv2

    cleaned = cv2.bitwise_not(inv)
    if fill_value == 0:
        return cleaned
    # optional polarity control
    return np.where(cleaned > 0, fill_value, 0).astype(np.uint8)

def morph_clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _adaptive_block_size(gray):
    h, w = gray.shape[:2]
    bs = int(max(15, min(75, round(min(h, w) / 30))))
    return bs + 1 if bs % 2 == 0 else bs

def mask_adaptive(gray, invert, cval=10):
    typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    block = _adaptive_block_size(gray)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, typ, block, cval)
    return morph_clean(clear_border(floodfill_background(m)))

def mask_otsu(gray, invert):
    if invert:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morph_clean(clear_border(floodfill_background(m)))

def watershed_split(mask, fg_ratio=0.45):
    if mask is None or mask.size == 0 or mask.max() == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    out = np.zeros_like(mask)
    out[markers > 1] = 255
    return out

def robust_area_bounds(cnts, img_area, small_batch):
    g_min = int(img_area * 0.0008)
    g_max = int(img_area * 0.18)
    if len(cnts) < 15:
        return g_min, g_max
    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    areas = areas[areas > img_area * 0.0004]
    if areas.size == 0:
        return g_min, g_max
    p20, p80 = np.percentile(areas, [20, 80])
    min_a = max(int(p20 * 0.5), g_min)
    max_a = min(int(p80 * 2.5), g_max)
    if min_a >= max_a:
        min_a, max_a = g_min, g_max
    return min_a, max_a

def filter_contours(cnts, img_area, small_batch):
    min_a, max_a = robust_area_bounds(cnts, img_area, small_batch)
    min_solidity = 0.82 if small_batch else 0.80
    max_ar       = 3.0  if small_batch else 4.0
    kept = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        if a / ha < min_solidity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar > max_ar:
            continue
        kept.append(c)
    if not kept and cnts:
        cnts_sorted = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        kept = cnts_sorted[: min(5, len(cnts_sorted))]
    return kept

def quality_score(cnts):
    if not cnts:
        return 0.0
    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    m, s = float(areas.mean()), float(areas.std())
    if m <= 1:
        return 0.0
    cv = s / m
    cv_term = 1.0 / (1.0 + cv)                 # lower spread is better
    n_term  = 1.0 - math.exp(-len(cnts) / 10.0)
    return 0.6 * cv_term + 0.4 * n_term

def process_one_mask(mask, small_batch):
    mask = watershed_split(mask, fg_ratio=0.42 if small_batch else 0.48)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = mask.shape[0] * mask.shape[1]
    kept = filter_contours(cnts, img_area, small_batch)
    return kept, quality_score(kept)

def draw_result(bgr, cnts, count):
    out = bgr.copy()
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 3)
    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------- Pipeline ----------
if file_to_use is not None:
    pil = _safe_open_image(file_to_use)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1800)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illum_norm(gray)
    gray = tophat_enhance(gray)  # suppress soft shadows, emphasize pill bodies

    # Build several masks independently; evaluate each rather than OR-merge.
    candidates = []
    for _ in range(2):  # two passes gives slight diversity without bloat
        candidates.append(mask_adaptive(gray, invert=True,  cval=10))
        candidates.append(mask_adaptive(gray, invert=False, cval=10))
    candidates.append(mask_otsu(gray, invert=True))
    candidates.append(mask_otsu(gray, invert=False))

    # Heuristic: if many pixels are foreground across masks, treat as large-batch
    nz = [int(np.count_nonzero(m)) for m in candidates]
    small_batch = np.median(nz) < (gray.size * 0.10) if len(nz) else True

    results = []
    for m in candidates:
        kept, score = process_one_mask(m.copy(), small_batch=small_batch)
        results.append((kept, score))

    kept_best = []
    if any(len(k) > 0 for k, _ in results):
        kept_best, _ = max(results, key=lambda r: r[1])
    else:
        kept_best = max(results, key=lambda r: len(r[0]))[0] if results else []

    count = len(kept_best)
    rgb = draw_result(bgr, kept_best, count)
    st.image(rgb, caption="Detected pills with count overlay", use_container_width=True)
    st.success(f"Pill count: {count}")

    if count == 0:
        st.info("No pills detected. Try a plain background, reduce glare, or space pills slightly and retake.")
else:
    st.info("Upload an image or use the camera to begin.")
