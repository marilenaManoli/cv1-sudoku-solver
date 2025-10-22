# Quick summary of the full flow
# find_edges() → detect grid edges.
# highlight_edges() → connect broken lines (optional).
# find_contours() → get all contours from edges.
# get_max_contour() → pick the largest (the Sudoku grid).
# find_corners() → approximate contour → 4 corner points.
# order_corners() → reorder to TL, TR, BR, BL.
# frontalize_image() → warp grid to a perfect square.

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
    # Typically you blur the image first to reduce noise,
    # then run cv2.Canny() with two thresholds.
    # Example: edges = cv2.Canny(image, threshold1=50, threshold2=150)
    # Experiment with the thresholds to make grid lines clear.

    # BEGIN YOUR CODE
    # edges = ...
    # return edges
    # END YOUR CODE

    raise NotImplementedError


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
    # Use morphological operations (dilate + close) with small kernels
    # to strengthen the grid boundaries.
    # Example:
    # kernel = np.ones((3,3), np.uint8)
    # highlighted_edges = cv2.dilate(edges, kernel, iterations=1)
    # highlighted_edges = cv2.morphologyEx(highlighted_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # BEGIN YOUR CODE
    # highlighted_edges = ...
    # return highlighted_edges
    # END YOUR CODE

    raise NotImplementedError


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
    # Use cv2.findContours with cv2.RETR_EXTERNAL to get outer contours only.
    # Example:
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # BEGIN YOUR CODE
    # contours = ...
    # return contours
    # END YOUR CODE

    raise NotImplementedError


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
    # You can use cv2.contourArea to compute the area for each contour.
    # Example:
    # max_contour = max(contours, key=cv2.contourArea)

    # BEGIN YOUR CODE
    # max_contour = ...
    # return max_contour
    # END YOUR CODE

    raise NotImplementedError


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
    # You can order by the sum (x+y) and difference (x-y):
    #  - top-left has smallest (x+y)
    #  - bottom-right has largest (x+y)
    #  - top-right has smallest (x-y)
    #  - bottom-left has largest (x-y)
    #
    # Example:
    # s = corners.sum(axis=1)
    # d = np.diff(corners, axis=1)
    # tl = corners[np.argmin(s)]
    # br = corners[np.argmax(s)]
    # tr = corners[np.argmin(d)]
    # bl = corners[np.argmax(d)]

    # BEGIN YOUR CODE
    # top_left = ...
    # top_right = ...
    # bottom_right = ...
    # bottom_left = ...
    # ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])
    # return ordered_corners
    # END YOUR CODE

    raise NotImplementedError


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
    # Use cv2.arcLength(contour, True) to get the contour perimeter,
    # then cv2.approxPolyDP(contour, epsilon * perimeter, True)
    # to simplify it into a polygon.
    #
    # If it doesn’t give exactly 4 corners, fake some to avoid crashing.

    # BEGIN YOUR CODE
    # corners = ...
    # if len(corners) != 4:
    #     corners += np.array([[0,0], [0,1], [1,0], [1,1]])
    #     corners = corners[:4]
    # ordered_corners = order_corners(corners)
    # return ordered_corners
    # END YOUR CODE

    raise NotImplementedError


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
    # Use cv2.resize(image, None, fx=scale, fy=scale)
    # Keep the same interpolation (e.g. cv2.INTER_AREA).

    # BEGIN YOUR CODE
    # rescaled_image = ...
    # return rescaled_image
    # END YOUR CODE

    raise NotImplementedError


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
    # Use cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    # Example: blurred_image = cv2.GaussianBlur(image, (5,5), sigma)

    # BEGIN YOUR CODE
    # blurred_image = ...
    # return blurred_image
    # END YOUR CODE

    raise NotImplementedError


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
    # Use np.linalg.norm(point1 - point2)
    # BEGIN YOUR CODE
    # distance = ...
    # return distance
    # END YOUR CODE

    raise NotImplementedError


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

    # The side length of the warped image can be the maximum of
    # the distances between opposite sides.
    # Example:
    # side = int(max(distance(tl,tr), distance(tr,br),
    #                distance(br,bl), distance(bl,tl)))

    # The destination points define a perfect square of that side length.

    # BEGIN YOUR CODE
    # side = ...
    # destination_points = np.array([[0,0], [side-1,0], [side-1,side-1], [0,side-1]], dtype=np.float32)
    # transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)
    # warped_image = cv2.warpPerspective(image, transform_matrix, (side, side))
    # assert warped_image.shape[0] == warped_image.shape[1], "height and width must match"
    # return warped_image
    # END YOUR CODE

    raise NotImplementedError


# ---------------------------------------------------------------------
# Utility: visualize all frontalized images
# ---------------------------------------------------------------------
def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    """
    Runs the full pipeline on a list of image paths and displays
    the resulting frontalized Sudoku grids in a grid of subplots.
    """
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, _ = pipeline(sudoku_image)
        show_image(frontalized_image, axis=axis, as_gray=True)
