import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# ---------------------- Helpers ----------------------
def downscale(img_bgr, max_dim=1600):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def apply_gamma(rgb, gamma=1.0):
    if gamma <= 0: gamma = 1.0
    inv = 1.0 / gamma
    table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(rgb, table)

def preprocess(bgr, gamma=1.1):
    # Light normalization
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = apply_gamma(rgb, gamma=gamma)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray

def segment(gray, block_size=21, c_val=10):
    block_size = int(max(15, block_size) // 2 * 2 + 1)  # odd >= 15
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, c_val
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def split_touching(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_bg = cv2.dilate(mask, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    sep = np.zerosLike(mask:=mask) if False else np.zeros_like(mask)  # compatibility
    sep[markers > 1] = 255
    return sep

def contour_filters(cnt, img_area, min_area_px, max_area_frac=0.12,
                    min_solidity=0.82, min_ar=0.35, max_ar=3.0):
    area = cv2.contourArea(cnt)
    if area < min_area_px or area > img_area * max_area_frac:
        return False
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area
    if solidity < min_solidity:
        return False
    x,y,w,h = cv2.boundingRect(cnt)
    ar = w / float(h) if h > 0 else 1.0
    if not (min_ar <= ar <= max_ar) and not (min_ar <= 1/ar <= max_ar):
        return False
    return True

def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    # 1.0 = perfect circle, lower = more irregular/elongated
    return 4.0 * math.pi * area / (peri * peri)

# ---------------------- UI: Controls ----------------------
with st.expander("Advanced (optional)"):
    st.write("If counts look wrong, tweak these and retake the photo.")
    blk = st.slider("Adaptive block size", 17, 59, 23, step=2)
    cval = st.slider("Adaptive C", -10, 20, 10, step=1)
    min_area_mult = st.slider("Min area (× image %)", 0.002, 0.02, 0.006, step=0.001)
    show_masks = st.checkbox("Show debug masks")

round_only = st.toggle("Round pills only", value=False)
if round_only:
    circ_thresh = st.slider("Roundness threshold", 0.60, 0.95, 0.78, step=0.01)
    max_ellipse_ar = st.slider("Max round pill aspect ratio", 1.0, 2.0, 1.35, step=0.01)

# ---------------------- Pipeline ----------------------
if file_to_use is not None:
    pil_img = Image.open(file_to_use).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1600)

    gray = preprocess(bgr, gamma=1.1)
    mask = segment(gray, block_size=int(blk), c_val=int(cval))
    mask = split_touching(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = mask.shape[0] * mask.shape[1]
    min_area_px = int(img_area * float(min_area_mult))

    filtered = []
    for c in contours:
        if not contour_filters(c, img_area, min_area_px=min_area_px):
            continue
        if round_only:
            # Apply circularity and near-equal axes check
            circ = circularity(c)
            x,y,w,h = cv2.boundingRect(c)
            ar = max(w, h) / max(1, min(w, h))  # >= 1
            if circ < circ_thresh or ar > max_ellipse_ar:
                continue
        filtered.append(c)

    out = bgr.copy()
    color = (0,0,255)  # red outlines
    cv2.drawContours(out, filtered, -1, color, 3)
    count = len(filtered)

    cv2.putText(out, f"Pill Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    st.image(out, caption="Detected pills with count overlay", use_column_width=True)
    st.success(f"Pill count: {count}")

    if show_masks:
        st.subheader("Debug views")
        st.image(gray, caption="Gray", use_column_width=True, clamp=True)
        st.image(mask, caption="Final mask (post-watershed)", use_column_width=True)
