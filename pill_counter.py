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
def downscale(bgr, max_dim=1800):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def illum_norm(gray):
    # flatten slow lighting gradients
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

def loose_components(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def robust_area_bounds(cnts, img_area):
    """Pick min/max contour area using percentiles with safe global clamps.
       This prevents 'all filtered out' failures."""
    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    areas = areas[areas > img_area * 0.0004]  # ignore dust
    if areas.size == 0:
        # very loose fallback
        return int(img_area*0.001), int(img_area*0.25)
    p25, p50, p75 = np.percentile(areas, [25, 50, 75])
    # Use interquartile scale; allow capsules and small tablets
    min_a = max(int(p25 * 0.40), int(img_area * 0.0008))
    max_a = min(int(p75 * 3.0),  int(img_area * 0.20))
    if min_a >= max_a:  # safety
        min_a = int(img_area*0.001)
        max_a = int(img_area*0.20)
    return min_a, max_a

def filter_pills(cnts, img_area, min_a, max_a):
    keep = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        # solidity to drop tray edges / ragged stuff
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        if a / ha < 0.80:
            continue
        # aspect ratio: allow round/oval/capsule, reject strips
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w,h)/max(1,min(w,h))
        if ar > 4.0:
            continue
        keep.append(c)
    return keep

def add_round_backup(gray, current_cnts, img_area, min_a, max_a):
    # Hough on edges; adds obvious round tablets missed by contour path
    edges = cv2.Canny(gray, 60, 160)
    try:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18,
                                   param1=120, param2=18, minRadius=6, maxRadius=60)
    except Exception:
        circles = None
    if circles is None:
        return current_cnts
    added = []
    for x, y, r in np.uint16(np.around(circles))[0, :]:
        area = math.pi * (int(r)**2)
        if area < min_a or area > max_a:
            continue
        # avoid duplicates (inside an existing contour)
        inside = any(cv2.pointPolygonTest(k, (int(x), int(y)), False) >= 0 for k in current_cnts)
        if inside:
            continue
        c = cv2.ellipse2Poly((int(x), int(y)), (int(r), int(r)), 0, 0, 360, 6)
        added.append(c.reshape(-1,1,2))
    return current_cnts + added

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

    # Build 4 simple masks and OR them
    m1 = mask_adaptive(gray, invert=True)    # light pills on dark
    m2 = mask_adaptive(gray, invert=False)   # dark pills on light
    m3 = mask_otsu(gray, invert=True)
    m4 = mask_otsu(gray, invert=False)
    merged = cv2.bitwise_or(cv2.bitwise_or(m1, m2), cv2.bitwise_or(m3, m4))
    merged = morph(merged)

    # Split touching gently
    split = watershed_split(merged, fg_ratio=0.45)

    # Loose components to learn typical pill area
    c_loose, _ = cv2.findContours(split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = split.shape[0] * split.shape[1]
    min_a, max_a = robust_area_bounds(c_loose, img_area)

    # Final filter
    cnts = filter_pills(c_loose, img_area, min_a, max_a)

    # Backup: add obvious round tablets missed
    cnts = add_round_backup(gray, cnts, img_area, min_a, max_a)

    count = len(cnts)
    rgb = draw(bgr, cnts, count)

    st.image(rgb, caption="Detected pills with count overlay", use_container_width=True)
    st.success(f"Pill count: {count}")

    if count == 0:
        st.info("No pills detected. Try a plain background and reduce glare, then retake one more photo.")
