import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# ---------- Page ----------
st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# ---------- Helpers ----------
def downscale(bgr, max_dim=1600):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def illum_norm(gray):
    # flatten slow lighting gradients (fluorescent/stainless)
    blur = cv2.GaussianBlur(gray, (0,0), 33)
    blur = np.clip(blur, 1, 255)
    norm = (gray.astype(np.float32) / blur.astype(np.float32)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def clear_border(mask, pct=0.02):
    # zero out a thin border so the frame/tray edge never becomes a giant contour
    h, w = mask.shape[:2]
    m = int(round(min(h, w) * pct))
    mask[:m, :] = 0
    mask[-m:, :] = 0
    mask[:, :m] = 0
    mask[:, -m:] = 0
    return mask

def morph_clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def mask_adaptive(gray, invert, block=23, cval=10):
    typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, typ, block, cval)
    return morph_clean(clear_border(m))

def mask_otsu(gray, invert):
    if invert:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morph_clean(clear_border(m))

def remove_giant_regions(mask, frac=0.35):
    # drop components that are unrealistically large (background chunks)
    img_area = mask.shape[0] * mask.shape[1]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > img_area * frac:
            cv2.drawContours(mask, [c], -1, 0, -1)
    return mask

def watershed_split(mask, fg_ratio=0.45):
    if mask.max() == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_bg = cv2.dilate(mask, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    out = np.zeros_like(mask)
    out[markers > 1] = 255
    return out

def filter_contours(cnts, img_area, small_batch):
    # Wide, safe gates. Slightly stricter shapes for small batches.
    min_area = int(img_area * (0.0006 if small_batch else 0.0005))
    max_area = int(img_area * (0.10   if small_batch else 0.15))

    min_solidity = 0.82 if small_batch else 0.80
    max_ar       = 3.0  if small_batch else 4.0

    kept = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        if a / ha < min_solidity:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w, h) / max(1, min(w, h))
        if ar > max_ar:
            continue
        kept.append(c)

    # Safety net: if we had candidates but filtered to zero, keep a few largest
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
    cv_term = 1.0 / (1.0 + cv)                 # 0..1 lower is better
    n_term  = 1.0 - math.exp(-len(cnts)/10.0)  # 0..1 more items -> more confidence
    return 0.6 * cv_term + 0.4 * n_term

def process_one_mask(mask, small_batch):
    mask = remove_giant_regions(mask)
    mask = watershed_split(mask, fg_ratio=0.42 if small_batch else 0.48)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = mask.shape[0] * mask.shape[1]
    kept = filter_contours(cnts, img_area, small_batch)
    return kept, quality_score(kept)

def draw_result(bgr, cnts, count):
    out = bgr.copy()
    cv2.drawContours(out, cnts, -1, (0,0,255), 3)
    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------- Pipeline ----------
if file_to_use is not None:
    pil = Image.open(file_to_use).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1600)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illum_norm(gray)

    # Build several simple masks (no OR-merging). We will process each separately and pick the best.
    candidates = []
    for bs in (19, 23, 27):
        candidates.append(mask_adaptive(gray, invert=True,  block=bs, cval=10))
        candidates.append(mask_adaptive(gray, invert=False, block=bs, cval=10))
    candidates.append(mask_otsu(gray, invert=True))
    candidates.append(mask_otsu(gray, invert=False))

    # Small-vs-large heuristic from the least-empty mask
    nonzero_counts = [int(np.count_nonzero(m)) for m in candidates]
    small_batch = True
    if nonzero_counts:
        # if there is a ton of foreground pixels in many masks, assume large batch
        small_batch = np.median(nonzero_counts) < (gray.size * 0.10)

    results = []
    for m in candidates:
        kept, score = process_one_mask(m.copy(), small_batch=small_batch)
        results.append((kept, score))

    # Choose the mask with the highest quality score. If all zero, choose the one with most contours.
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
