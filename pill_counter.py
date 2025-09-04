import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import math

# -------------------- UI --------------------
st.set_page_config(page_title="Pill Counter (from scratch)", page_icon="💊", layout="centered")
st.title("Pill Counter")

uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg","jpeg","png"])
camera_file   = st.camera_input("Or take a picture with your camera")
file_to_use = uploaded_file or camera_file

# -------------------- Core helpers --------------------
def safe_open(file):
    pil = Image.open(file)
    pil = ImageOps.exif_transpose(pil)   # fix phone EXIF rotation
    pil = pil.convert("RGB")
    return np.array(pil)

def downscale(bgr, max_dim=1600):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def to_lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

def mahala_bg_mask(lab, border_frac=0.05, chi2_thresh=11.34, eps=5.0):
    """
    Learn background color from border pixels (largest, most reliable sample),
    then keep everything whose Mahalanobis distance to that color is *large* (foreground).
    chi2_thresh ~ 99th pct for df=3 (L,a,b).
    eps is Tikhonov regularization to stabilize cov inverse.
    """
    h, w = lab.shape[:2]
    bf = int(max(2, round(min(h, w) * border_frac)))
    border = np.vstack([
        lab[:bf, :, :].reshape(-1, 3),
        lab[-bf:, :, :].reshape(-1, 3),
        lab[:, :bf, :].reshape(-1, 3),
        lab[:, -bf:, :].reshape(-1, 3),
    ])
    mu = border.mean(axis=0)  # (3,)
    X = border - mu
    cov = (X.T @ X) / max(1, len(X) - 1)
    cov += np.eye(3) * eps
    cov_inv = np.linalg.inv(cov)

    # Mahalanobis distance^2 for each pixel
    L = lab.reshape(-1,3).astype(np.float32) - mu
    d2 = np.einsum("ij,jk,ik->i", L, cov_inv, L).reshape(h, w)

    fg = (d2 > chi2_thresh).astype(np.uint8) * 255
    # Clean: kill tiny specks, fill tiny holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    # Remove anything touching corners (safety against border bleed)
    inv = cv2.bitwise_not(fg)
    ffmask = np.zeros((h+2, w+2), dtype=np.uint8)
    for seed in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
        ffmask[:] = 0
        cv2.floodFill(inv, ffmask, seedPoint=(seed[1], seed[0]), newVal=0)
    fg = cv2.bitwise_not(inv)
    return fg

def white_core(gray, mask_fg):
    """
    Keep bright pill tops inside the foreground (reject shadows).
    Top-hat highlights domed tops; Otsu does the rest.
    """
    h, w = gray.shape[:2]
    k = int(max(9, min(41, round(min(h, w)/28))))
    if k % 2 == 0: k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    # Constrain to foreground to avoid background speculars
    masked = cv2.bitwise_and(tophat, tophat, mask=mask_fg)
    _, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # clean small noise
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=1)
    return bw

def estimate_pill_radius(core_mask):
    """
    Estimate typical radius from bright cores via equivalent radius.
    Robust median; clamp to [6, 120] pixels.
    """
    num, lbl = cv2.connectedComponents(core_mask)
    radii = []
    for i in range(1, num):
        area = int((lbl == i).sum())
        if area < 15: 
            continue
        r_eq = math.sqrt(area / math.pi)
        radii.append(r_eq)
    if not radii:
        return 18  # reasonable default for ~1600px side image
    r = float(np.median(radii))
    return float(max(6, min(120, r)))

def local_maxima_seeds(fg_mask, min_peak_dist, min_peak_strength_frac=0.25):
    """
    Seeds = local maxima of distance transform inside the foreground.
    Enforce *spacing* via morphological NMS with window ~ min_peak_dist.
    """
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    ksize = max(3, int(round(min_peak_dist))*2 + 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    local_max = (dist == cv2.dilate(dist, k))
    strong = dist > (dist.max() * float(min_peak_strength_frac))
    peaks = np.logical_and(local_max, strong).astype(np.uint8)*255

    # Turn peaks into markers
    num, markers = cv2.connectedComponents(peaks)
    # If somehow no peaks survived, fall back to a single seed at global max
    if num <= 1 and dist.max() > 0:
        peaks = np.zeros_like(fg_mask, dtype=np.uint8)
        yx = np.unravel_index(np.argmax(dist), dist.shape)
        peaks[yx] = 255
        num, markers = cv2.connectedComponents(peaks)
    return num, markers, dist

def split_by_watershed(fg_mask, r_typ):
    """
    Split clumps using distance-based watershed driven by seed spacing ~ r_typ.
    """
    # Slight erode ensures seeds are interior to blobs
    er_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg_er = cv2.erode(fg_mask, er_k, iterations=1)

    num, markers, dist = local_maxima_seeds(fg_er, min_peak_dist=r_typ*0.8, min_peak_strength_frac=0.20)
    sure_bg = cv2.dilate(fg_er, er_k, iterations=2)
    unknown = cv2.subtract(sure_bg, (markers>0).astype(np.uint8)*255)

    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    seg = np.zeros_like(fg_mask)
    seg[markers > 1] = 255
    return seg, markers

def filter_and_count(segmentation, r_typ):
    """
    Keep components that look like an individual pill:
      - area within [0.45, 2.2] * area_typ  (we'll split >1.8 below)
      - shape not too elongated (aspect ratio <= 3)
      - good solidity
    Also: re-split any big leftover blobs using a focused watershed inside them.
    """
    area_typ = math.pi * (r_typ ** 2)
    min_a = 0.45 * area_typ
    max_a = 2.2  * area_typ

    final_mask = np.zeros_like(segmentation)
    num, lbl = cv2.connectedComponents(segmentation)

    kept = 0
    for i in range(1, num):
        comp = (lbl == i).astype(np.uint8)*255
        a = comp.sum()
        if a < 10: 
            continue

        # If too large, try to split internally into multiples of area_typ
        if a > 1.8 * area_typ:
            # Restrict watershed to this component
            seg_sub = comp.copy()
            _, markers, dist = local_maxima_seeds(seg_sub, min_peak_dist=r_typ*0.8, min_peak_strength_frac=0.18)
            color = cv2.cvtColor(seg_sub, cv2.COLOR_GRAY2BGR)
            mk = markers + 1
            mk[seg_sub==0] = 0
            mk = cv2.watershed(color, mk)
            sub = np.zeros_like(seg_sub); sub[mk > 1] = 255

            # Count sub-components that match individual size
            n2, l2 = cv2.connectedComponents(sub)
            for j in range(1, n2):
                subc = (l2 == j).astype(np.uint8)*255
                aa = subc.sum()
                if min_a <= aa <= max_a:
                    final_mask = cv2.bitwise_or(final_mask, subc)
                    kept += 1
            continue

        # For plausible singles, check shape gates
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        area = cv2.contourArea(c)
        ha = cv2.contourArea(hull) + 1e-6
        solidity = area / ha
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w,h) / max(1, min(w,h))  # aspect ratio

        if (min_a <= area <= max_a) and (solidity >= 0.80) and (ar <= 3.0):
            final_mask = cv2.bitwise_or(final_mask, comp)
            kept += 1

    return final_mask, kept

def draw_count(bgr, mask, count):
    out = bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0,0,255), 3)
    cv2.putText(out, f"Pill Count: {count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# -------------------- Pipeline --------------------
if file_to_use is not None:
    rgb = safe_open(file_to_use)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = downscale(bgr, 1600)

    lab = to_lab(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) Identify & remove background (pure color model from borders)
    fg_mask = mahala_bg_mask(lab)

    # 2) Keep bright pill tops only (shadow-resistant)
    core = white_core(gray, fg_mask)

    # 3) Estimate a typical pill size from cores
    r_typ = estimate_pill_radius(core)

    # 4) Split clumps in the full foreground using r_typ spacing
    #    (we ignore splitting accuracy for a moment; focus on singles)
    seg, markers = split_by_watershed(fg_mask, r_typ)

    # 5) Keep components that look like single pills; split big leftovers into multiples
    singles_mask, count = filter_and_count(seg, r_typ)

    # 6) Display
    out = draw_count(bgr, singles_mask, count)
    st.image(out, caption=f"Detected single pills (r_typ≈{r_typ:.1f}px).", use_container_width=True)
    st.success(f"Pill count: {count}")

    if count == 0:
        st.info("No pills detected. If the background color is similar to pills, try a darker backdrop or stronger lighting.")
else:
    st.info("Upload an image or use the camera to begin.")
