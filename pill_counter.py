import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import math

# ================= UI =================
st.set_page_config(page_title="Smart Pill Counter (Min 8)", page_icon="💊", layout="centered")
st.title("Smart Pill Counter")

up = st.file_uploader("Upload a pill tray image", type=["jpg","jpeg","png"])
cam = st.camera_input("Or take a picture with your camera")
file_to_use = up or cam

# ================ Utils ===============
def safe_open(file):
    pil = Image.open(file); pil = ImageOps.exif_transpose(pil); pil = pil.convert("RGB")
    return np.array(pil)

def downscale(bgr, max_dim=1700):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim: return bgr
    s = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def clear_border(mask, pct=0.02):
    h, w = mask.shape[:2]; m = int(round(min(h, w) * pct))
    if m < 1: return mask
    mask[:m,:]=0; mask[-m:,:]=0; mask[:,:m]=0; mask[:,-m:]=0
    return mask

def floodfill_corners(mask):
    inv = cv2.bitwise_not(mask); h, w = mask.shape[:2]
    ff = np.zeros((h+2, w+2), dtype=np.uint8)
    for seed in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
        ff[:] = 0; cv2.floodFill(inv, ff, (seed[1], seed[0]), 0)
    return cv2.bitwise_not(inv)

def morph_clean(mask, open_iter=1, close_iter=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return mask

def to_lab(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

def illum_norm(gray):
    h, w = gray.shape[:2]
    sigma = max(7, min(45, round(min(h, w)/24)))
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma, sigmaY=sigma)
    blur = np.clip(blur, 1, 255)
    norm = (gray.astype(np.float32)/blur.astype(np.float32))*255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

# ====== 1) BACKGROUND MODEL ======
def mahala_fg_from_borders(lab, border_frac=0.06, chi2=11.34, eps=6.0):
    h, w = lab.shape[:2]
    b = int(max(2, round(min(h,w)*border_frac)))
    border = np.vstack([
        lab[:b,:,:].reshape(-1,3), lab[-b:,:,:].reshape(-1,3),
        lab[:,:b,:].reshape(-1,3), lab[:,-b:,:].reshape(-1,3),
    ]).astype(np.float32)
    mu = border.mean(axis=0); X = border - mu
    cov = (X.T @ X) / max(1, len(X)-1); cov += np.eye(3)*eps
    cov_inv = np.linalg.inv(cov)
    L = lab.reshape(-1,3).astype(np.float32) - mu
    d2 = np.einsum("ij,jk,ik->i", L, cov_inv, L).reshape(h, w)
    fg = (d2 > chi2).astype(np.uint8)*255
    fg = clear_border(fg, pct=0.02); fg = floodfill_corners(fg); fg = morph_clean(fg,1,2)
    return fg

# ====== 2) BRIGHT CORES ======
def bright_cores(gray, fg):
    h, w = gray.shape[:2]
    k = int(max(9, min(41, round(min(h,w)/28))));  k += (k%2==0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    masked = cv2.bitwise_and(tophat, tophat, mask=fg)
    _, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morph_clean(bw, 1, 1)

def estimate_r_typ(core):
    n, lab = cv2.connectedComponents(core); radii=[]
    for i in range(1, n):
        area=int((lab==i).sum())
        if area<15: continue
        radii.append(math.sqrt(area/math.pi))
    if not radii: return None
    r=float(np.median(radii)); return max(6.0, min(120.0, r))

# ====== 3) EXTRA MASKS ======
def adaptive_mask(gray, fg, cval=8):
    h,w=gray.shape[:2]
    bs=int(max(15, min(75, round(min(h,w)/30)))); bs+= (bs%2==0)
    m1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bs,cval)
    m2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,cval)
    m=cv2.bitwise_or(m1,m2); m=cv2.bitwise_and(m,fg)
    return morph_clean(clear_border(m),1,2)

def otsu_mask(gray, fg):
    _, b = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, i = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    m=cv2.bitwise_or(b,i); m=cv2.bitwise_and(m,fg)
    return morph_clean(clear_border(m),1,2)

# ====== 4) SEEDS ======
def localmax_seeds(mask, r_typ, frac=0.18):
    dist=cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ksize=max(5, int(round(r_typ*1.2))|1)
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))
    local=(dist==cv2.dilate(dist,k)); strong=dist>(dist.max()*frac)
    peaks=(np.logical_and(local,strong)).astype(np.uint8)*255
    n, markers=cv2.connectedComponents(peaks)
    return n, markers, dist

def hough_circle_seeds(gray, fg, r_typ):
    m=cv2.bitwise_and(gray,gray,mask=fg); m=cv2.GaussianBlur(m,(0,0),1.2)
    r=int(round(max(6, min(120, r_typ))))
    circles=cv2.HoughCircles(m, cv2.HOUGH_GRADIENT, dp=1.2, minDist=r*0.9,
                             param1=120, param2=18,
                             minRadius=int(r*0.6), maxRadius=int(r*1.6))
    seeds=np.zeros_like(fg)
    if circles is not None:
        for x,y,rr in np.round(circles[0,:]).astype(int):
            if 0<=x<fg.shape[1] and 0<=y<fg.shape[0] and fg[y,x]>0:
                seeds[y,x]=255
    n, markers=cv2.connectedComponents(seeds)
    return n, markers

# ====== 5) WATERSHED ======
def watershed_split(mask, seeds_markers):
    if mask.max()==0: return mask
    mk=seeds_markers.copy(); mk=mk+1; mk[mask==0]=0
    color=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mk=cv2.watershed(color, mk)
    seg=np.zeros_like(mask); seg[mk>1]=255
    return seg

# ====== 6) FILTER + COUNT ======
def filter_count(seg, r_typ):
    area_typ=math.pi*(r_typ**2)
    min_a, max_a = 0.45*area_typ, 2.2*area_typ
    final=np.zeros_like(seg); kept=0
    n,lab=cv2.connectedComponents(seg)
    for i in range(1,n):
        comp=(lab==i).astype(np.uint8)*255
        cnts,_=cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c=max(cnts,key=cv2.contourArea); a=cv2.contourArea(c)
        if a<10: continue
        hull=cv2.convexHull(c); ha=cv2.contourArea(hull)+1e-6
        solidity=a/ha; x,y,w,h=cv2.boundingRect(c)
        ar=max(w,h)/max(1,min(w,h)); peri=cv2.arcLength(c,True)+1e-6
        circ=4*math.pi*a/(peri*peri)
        if (min_a<=a<=max_a) and (solidity>=0.80) an
