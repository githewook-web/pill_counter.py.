import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------- Page ----------
st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# ---------- Helpers ----------
def downscale(bgr, max_dim=1800):
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

def morph(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def mask_adaptive(gray, invert):
    typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, typ, 23, 10)
    return morph(m)

def mask_otsu(gray, invert):
    if invert:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morph(m)

def watershed_split(mask, fg_ratio):
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

def robust_area_bounds(cnts, img_area, small_batch):
    # Global clamps (prevent "filter to zero")
    g_min = int(img_area * 0.0008)   # very small tablet at phone distance
    g_max = int(img_area * 0.18)     # avoid tray edges / huge blobs

    if len(cnts) < 15:
        # Small-batch bias: be generous; rely on solidity/shape to reject junk
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

def filter_pills(cnts, img_area, min_a, max_a, small_batch):
    keep = []
    # Shape gates: slightly stricter for small batches
    min_solidity = 0.82 if small_batch else 0.80
    max_ar       = 3.0  if small_batch else 4.0  # aspect ratio (capsules ok, strips no)

    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        if a / ha < min_solidity:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w,h) / max(1, min(w,h))
        if ar > max_ar:
            continue
        keep.append(c)

    # Safety net: never return zero if there were candidates
    if not keep and cnts:
        cnts_sorted = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        keep = cnts_sorted[: min(5, len(cnts_sorted))]  # keep a few largest blobs
    return keep

def draw(bgr, cnts, count):
    out = bgr.copy()
    cv2.drawContours(out, cnts, -1, (0,0,255), 3)
    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------- Pipeline ----------
if file_to_use is not None:
    pil = Image.open(file_to_use).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1800)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illum_norm(gray)

    # Four simple masks (covers light/dark pills + different backgrounds)
    m1 = mask_adaptive(gray, invert=True)
    m2 = mask_adaptive(gray, invert=False)
    m3 = mask_otsu(gray, invert=True)
    m4 = mask_otsu(gray, invert=False)
    merged = cv2.bitwise_or(cv2.bitwise_or(m1, m2), cv2.bitwise_or(m3, m4))
    merged = morph(merged)

    # Watershed split (stronger split for small batches)
    # First estimate small vs large from raw components
    cnts_est, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_batch = len(cnts_est) <= 50
    fg_ratio = 0.42 if small_batch else 0.48
    split = watershed_split(merged, fg_ratio=fg_ratio)

    # Learn bounds, then filter
    cnts_loose, _ = cv2.findContours(split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = split.shape[0] * split.shape[1]
    min_a, max_a = robust_area_bounds(cnts_loose, img_area, small_batch)
    cnts = filter_pills(cnts_loose, img_area, min_a, max_a, small_batch)

    count = len(cnts)
    rgb = draw(bgr, cnts, count)

    st.image(rgb, caption="Detected pills with count overlay", use_container_width=True)
    st.success(f"Pill count: {count}")

    if count == 0:
        st.info("No pills detected. Try a plain background, reduce glare, or space pills slightly and retake.")
