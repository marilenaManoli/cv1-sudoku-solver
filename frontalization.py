import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import read_image, show_image


# =====================================================================
# These are your core helper functions for extracting the Sudoku grid
# from a raw image (photo).
# =====================================================================


# ---------------------------------------------------------------------
# 1. EDGE DETECTION (CANNY)
# ---------------------------------------------------------------------
def find_edges(image):
    """
    Purpose:
        Find the edges in a grayscale image using the Canny edge detector.

    Args:
        image (np.array): grayscale image of shape [H, W]
    Returns:
        edges (np.array): binary mask (white = edge, black = background)
    """
    
    # BEGIN YOUR CODE
    # image is grayscale uint8 (H, W)
    v = np.median(image)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(image, lower, upper, L2gradient=True)
    return edges
    # END YOUR CODE


# ---------------------------------------------------------------------
# 2. HIGHLIGHT EDGES (OPTIONAL MORPHOLOGY)
# ---------------------------------------------------------------------
def highlight_edges(edges):
    """
    Purpose:
        Connect broken grid lines and make edges more visible
        by applying dilation or morphological closing operations.

    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlighted_edges (np.array): improved binary mask
    """
    # BEGIN YOUR CODE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated
    # END YOUR CODE



# ---------------------------------------------------------------------
# 3. FIND ALL CONTOURS
# ---------------------------------------------------------------------
def find_contours(edges):
    """
    Purpose:
        Find all contours (continuous lines/boundaries) in the edge map.

    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours: list of arrays, each representing a contour
    """
    # BEGIN YOUR CODE
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    # END YOUR CODE



# ---------------------------------------------------------------------
# 4. FIND THE BIGGEST CONTOUR
# ---------------------------------------------------------------------
def get_max_contour(contours):
    """
    Purpose:
        From all detected contours, pick the one with the largest area,
        since the Sudoku grid should be the largest rectangular object.

    Args:
        contours (list): list of contour arrays
    Returns:
        max_contour (np.array): contour with the largest area
    """
    # BEGIN YOUR CODE
    if contours is None or len(contours) == 0:
        raise ValueError("No contours found")
    return max(contours, key=cv2.contourArea)
    # END YOUR CODE



# ---------------------------------------------------------------------
# 5. ORDER THE CORNERS
# ---------------------------------------------------------------------
def order_corners(corners):
    """
    Purpose:
        Arrange four corner points in a fixed order:
        [top-left, top-right, bottom-right, bottom-left].

    Args:
        corners (np.array): array of shape [4, 2]
    Returns:
        ordered_corners (np.array): same 4 points in fixed order
    """

    # BEGIN YOUR CODE
    c = np.array(corners, dtype=np.float32).reshape(-1, 2)
    s = c.sum(axis=1)              # x + y
    d = np.diff(c, axis=1).ravel() # x - y
    tl = c[np.argmin(s)]
    br = c[np.argmax(s)]
    tr = c[np.argmin(d)]
    bl = c[np.argmax(d)]
    ordered_corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered_corners
    # END YOUR CODE



# ---------------------------------------------------------------------
# 6. APPROXIMATE CONTOUR AS 4 CORNERS
# ---------------------------------------------------------------------
def find_corners(contour, epsilon=0.42):
    """
    Purpose:
        Approximate the largest contour to a polygon with 4 vertices
        (the corners of the Sudoku grid).

    Args:
        contour (np.array): contour points of shape [N, 1, 2]
        epsilon (float): approximation factor (fraction of contour perimeter)
    Returns:
        ordered_corners (np.array): [4, 2] points ordered TL, TR, BR, BL
    """
    # BEGIN YOUR CODE
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon * peri, True)

    # If not 4 points, sweep a range of epsilons to find 4
    if len(approx) != 4:
        for e in np.linspace(0.01, 0.10, 10):
            approx = cv2.approxPolyDP(contour, e * peri, True)
            if len(approx) == 4:
                break

    # Fallback to min-area rectangle if still not 4
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = box.astype(np.float32)

    corners = approx.reshape(-1, 2).astype(np.float32)
    return corners
    # END YOUR CODE


     
# ---------------------------------------------------------------------
# 7. RESCALE IMAGE (OPTIONAL)
# ---------------------------------------------------------------------
def rescale_image(image, scale=0.42):
    """
    Purpose:
        Resize the image by a constant factor, to speed up processing
        or normalize different input sizes.

    Args:
        image (np.array): input image
        scale (float): scale factor (e.g. 0.5 halves the size)
    Returns:
        rescaled_image (np.array): 8-bit image resized
    """
    # BEGIN YOUR CODE
    h, w = image.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    rescaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rescaled_image
    # END YOUR CODE



# ---------------------------------------------------------------------
# 8. GAUSSIAN BLUR
# ---------------------------------------------------------------------
def gaussian_blur(image, sigma):
    """
    Purpose:
        Slightly smooth the image to reduce noise before edge detection.

    Args:
        image (np.array): input grayscale image
        sigma (float): blur strength
    Returns:
        blurred_image (np.array): smoothed image
    """
    # BEGIN YOUR CODE
    k = max(3, int(2 * round(3 * max(0.1, sigma)) + 1))  # odd kernel size
    blurred_image = cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
    return blurred_image
    # END YOUR CODE



# ---------------------------------------------------------------------
# 9. EUCLIDEAN DISTANCE (HELPER)
# ---------------------------------------------------------------------
def distance(point1, point2):
    """
    Purpose:
        Compute Euclidean distance between two points.

    Args:
        point1, point2 (np.array): coordinate pairs (x, y)
    Returns:
        distance (float)
    """
    # BEGIN YOUR CODE
    p1 = np.asarray(point1, dtype=np.float32)
    p2 = np.asarray(point2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))
    # END YOUR CODE


# ---------------------------------------------------------------------
# 10. FRONTALIZE IMAGE (PERSPECTIVE WARP)
# ---------------------------------------------------------------------
def frontalize_image(image, ordered_corners):
    """
    Purpose:
        Warp the Sudoku region into a square (bird’s-eye) view
        using a perspective transform.

    Args:
        image (np.array): original input image
        ordered_corners (np.array): [TL, TR, BR, BL]
    Returns:
        warped_image (np.array): square frontalized Sudoku image
    """
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    w1 = distance(top_left, top_right)
    w2 = distance(bottom_left, bottom_right)
    h1 = distance(top_left, bottom_left)
    h2 = distance(top_right, bottom_right)
    side = int(max(w1, w2, h1, h2))

    dst = np.array([[0, 0],
                    [side - 1, 0],
                    [side - 1, side - 1],
                    [0, side - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), dst)
    warped_image = cv2.warpPerspective(image, M, (side, side))
    assert warped_image.shape[0] == warped_image.shape[1], "height and width must match"
    return warped_image
    # END YOUR CODE


# ---------------------------------------------------------------------
# Utility: visualize all frontalized images
# ---------------------------------------------------------------------
def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    """
    Runs the full pipeline on a list of image paths and displays
    the resulting frontalized Sudoku grids in a grid of subplots.
    """
    
    # BEGIN YOUR CODE
    from const import FRONTALIZED_IMAGES_PATH

    ncols = 4
    nrows = (len(image_paths) + ncols - 1) // ncols
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes[np.newaxis, ...]

    total_slots = nrows * ncols
    for j in range(len(image_paths), total_slots):
        ax = axes[j // ncols][j % ncols]
        ax.axis("off")

    os.makedirs(FRONTALIZED_IMAGES_PATH, exist_ok=True)

    for i, image_path in enumerate(tqdm(image_paths)):
        ax = axes[i // ncols][i % ncols]
        ax.set_title(os.path.split(image_path)[1])

        sudoku_image = read_image(image_path=image_path)

        if sudoku_image.ndim == 2:
            sudoku_image = cv2.cvtColor(sudoku_image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
        elif sudoku_image.ndim == 3 and sudoku_image.shape[2] == 1:
            sudoku_image = np.repeat(sudoku_image, 3, axis=2)
        elif sudoku_image.ndim == 3 and sudoku_image.shape[2] == 4:
            sudoku_image = sudoku_image[:, :, :3]

        frontalized_image, _ = pipeline(sudoku_image)

        show_image(frontalized_image, axis=ax, as_gray=True)

        out_path = os.path.join(FRONTALIZED_IMAGES_PATH, os.path.basename(image_path))
        if frontalized_image.dtype != np.uint8:
            frontalized_image = frontalized_image.astype(np.uint8)
        if frontalized_image.ndim == 3 and frontalized_image.shape[2] == 3:
            to_save = cv2.cvtColor(frontalized_image, cv2.COLOR_RGB2BGR)
        else:
            to_save = frontalized_image
        cv2.imwrite(out_path, to_save)
    # END YOUR CODE
