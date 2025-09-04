import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Smart Pill Counter")

# Upload OR Camera capture
uploaded_file = st.file_uploader("Upload a pill tray image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or take a picture with your camera")

# Pick whichever source was used
file_to_use = uploaded_file or camera_file

if file_to_use is not None:
    # Load image
    image = Image.open(file_to_use).convert("RGB")
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold (handles different lighting/pill colors)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny blobs
    pill_contours = [c for c in contours if cv2.contourArea(c) > 200]

    # Draw contours for preview
    output = img_array.copy()
    cv2.drawContours(output, pill_contours, -1, (255, 0, 0), 2)

    # Add pill count overlay text (always top-left corner)
    pill_count = len(pill_contours)
    cv2.putText(
        output,
        f"Pill Count: {pill_count}",
        (30, 50),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,       # Font scale
        (0, 255, 0),  # Green text
        3,         # Thickness
        cv2.LINE_AA
    )

    # Show results
    st.image(output, caption="Detected pills with count overlay")
    st.success(f"Pill count: {pill_count}")

