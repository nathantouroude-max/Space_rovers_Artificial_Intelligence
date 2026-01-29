import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# PARAMETERS
# =========================
IMAGE_DIR = Path(r"C:\Users\natha\Desktop\réinitialisation\IPSA\BAU\Artificial Intelligence\Report_project\images\render")  # <-- CHANGE THIS
DISPLAY_EVERY = 3000                                  # show 1 image every N images

MIN_OBSTACLE_AREA = 80                                # minimum contour area
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =========================
# OBSTACLE DETECTION
# =========================
def detect_obstacles(gray_image):
    """
    Detect obstacles using edge detection and contour filtering.
    Returns a list of contours.
    """
    blurred = cv2.GaussianBlur(gray_image, BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obstacles = [
        cnt for cnt in contours
        if cv2.contourArea(cnt) > MIN_OBSTACLE_AREA
    ]

    return obstacles


# =========================
# SAFE IMAGE LOADING (Windows / Unicode proof)
# =========================
def load_grayscale_image(image_path):
    """
    Robust image loading for Windows paths and Unicode characters.
    Returns None if the image cannot be loaded.
    """
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return image
    except Exception:
        return None


# =========================
# VISUALIZATION
# =========================
def display_detected_obstacles(image, obstacles, image_index):
    """
    Display the image with detected obstacles outlined.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(image_rgb, obstacles, -1, (255, 0, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(f"Detected obstacles (image #{image_index})")
    plt.axis("off")
    plt.show()


# =========================
# MAIN DATASET LOOP
# =========================
def process_dataset():
    image_files = sorted(IMAGE_DIR.iterdir())

    processed = 0

    for idx, image_path in enumerate(image_files):

        # Skip non-image files
        if not image_path.suffix.lower() in VALID_EXTENSIONS:
            continue

        image = load_grayscale_image(image_path)

        if image is None:
            print(f"❌ Could not load: {image_path.name}")
            continue

        obstacles = detect_obstacles(image)
        processed += 1

        if processed % DISPLAY_EVERY == 0:
            print(
                f"[INFO] Image {processed} | "
                f"File: {image_path.name} | "
                f"Obstacles detected: {len(obstacles)}"
            )
            display_detected_obstacles(image, obstacles, processed)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process_dataset()
