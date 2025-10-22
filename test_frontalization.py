import os, glob, cv2
import numpy as np
from pipeline import Pipeline
from const import TRAIN_IMAGES_PATH, FRONTALIZED_IMAGES_PATH
from utils import read_image, rgb2grayscale
from frontalization import (
    gaussian_blur, find_edges, highlight_edges,
    find_contours, get_max_contour, find_corners,
    order_corners, frontalize_image, show_frontalized_images
)

# --- ensure 3 channels so rgb2grayscale won't crash on grayscale inputs ---
def ensure_three_channels(image):
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image

# make sure the output folder exists
os.makedirs(FRONTALIZED_IMAGES_PATH, exist_ok=True)

# build the frontalization pipeline
frontalization_pipeline = Pipeline(
    functions=[
        ensure_three_channels,   # <— added
        rgb2grayscale,
        gaussian_blur,
        find_edges,
        highlight_edges,
        find_contours,
        get_max_contour,
        find_corners,
        order_corners,
        frontalize_image
    ],
    parameters={
        "gaussian_blur": {"sigma": 1.2},   # try 0.8–1.6 if needed
        "find_corners": {"epsilon": 0.02}  # try 0.01–0.06 if corners fail
    }
)

# pick a few images to test
image_paths = sorted(glob.glob(os.path.join(TRAIN_IMAGES_PATH, "*.jpg")))[:4]
# (optional) include PNGs too:
# image_paths = sorted(glob.glob(os.path.join(TRAIN_IMAGES_PATH, "*.jpg")))[:2] + \
#               sorted(glob.glob(os.path.join(TRAIN_IMAGES_PATH, "*.png")))[:2]

# run and visualize
show_frontalized_images(image_paths, pipeline=frontalization_pipeline, figsize=(14, 10))

# save the outputs
for path in image_paths:
    img = read_image(path)
    warped, _ = frontalization_pipeline(img)  # safe now (ensure_three_channels runs first)
    out_path = os.path.join(FRONTALIZED_IMAGES_PATH, os.path.basename(path))
    # ensure correct format for imwrite
    if warped.dtype != np.uint8:
        warped = warped.astype(np.uint8)
    if warped.ndim == 3 and warped.shape[2] == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, warped)
    print(f"Saved: {out_path}")
