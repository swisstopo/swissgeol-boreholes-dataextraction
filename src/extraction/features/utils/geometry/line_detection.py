"""Script for line detection in pdf pages."""

import logging
import os

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv
from extraction.features.utils.geometry.geometric_line_utilities import (
    merge_parallel_lines_quadtree,
)
from numpy.typing import ArrayLike
from utils.file_utils import read_params

from .geometry_dataclasses import Line
from .util import line_from_array

load_dotenv()

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

line_detection_params = read_params("line_detection_params.yml")


def detect_lines_lsd(page: pymupdf.Page, scale_factor=2, lsd_params=None) -> ArrayLike:
    """Given a file path, detect lines in the pdf using the Line Segment Detector (LSD) algorithm.

    Publication of the algorithm can be found here: http://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf
    Note: As of now the function only works for pdfs with a single page.
          For now the function displays each pdf with the lines detected using opencv.
          This behavior will be changed in the future.

    Args:
        page (pymupdf.Page): The page to detect lines in.
        scale_factor (float, optional): The scale factor to scale the pdf page. Defaults to 2.
        lsd_params (dict, optional): The parameters for the Line Segment Detector. Defaults to None.

    Returns:
        list[Line]: The lines detected in the pdf.
    """
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create default line segment detector
    lsd = cv2.createLineSegmentDetector(**lsd_params)
    #  Documentation for the parameters can be found here:
    #  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gae0bba3b867a5f44d1b823aef4f57ee8d

    # Detect lines in the image
    lines = lsd.detect(gray)[0]
    return [line_from_array(line, scale_factor) for line in lines]


def detect_lines_hough(page: pymupdf.Page, hough_params: dict) -> ArrayLike:
    """Given document, detect lines in the pdf using the hough transform algorithm.

    Args:
        page (pymupdf.Page): The page to detect lines in.
        hough_params (dict): The parameters for the bluring and hough transform.

    Returns:
        tuple[list[Line], float]: The lines detected in the pdf and the scale_ratio of the document
    """
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    target_width = hough_params["target_width"]
    h, w = gray.shape
    scale_ratio = target_width / w  # resize to a fixed size, to work with the chosen parameters
    new_size = (target_width, int(h * scale_ratio))  # carefull size is (w,h) for resizing
    gray = cv2.resize(gray, new_size)
    cv2.imwrite("data/debug_cv_gray.png", gray)

    blur_dist = hough_params["blur_dist"]
    blurred = cv2.bilateralFilter(gray, blur_dist, blur_dist * 5, blur_dist * 5)  # dist, sigma_color, sigma space
    cv2.imwrite("data/debug_cv_blur.png", blurred)

    # Apply binary thresholding with otsu method, that minimizes the intra-class variance.
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("data/debug_cv_otsu.png", thresh)

    morph_kernel_size = hough_params["morph_kernel_size"]
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    # morphed = cv2.morphologyEx(morphed, cv2.MORPH_DILATE, kernel)
    cv2.imwrite("data/debug_cv_morph.png", morphed)

    min_line_length = hough_params["min_line_length"]  # Minimum length of line (adjust based on your image)
    max_line_gap = hough_params["max_line_gap"]  # Maximum allowed gap between line segments
    rho = hough_params["rho"]  # Distance resolution in the accumulartor space in pixels
    theta = np.pi / 180 * hough_params["theta_deg"]  # Angular resolution in the accumulator space in radian
    threshold = hough_params["threshold"]  # Minimum number of votes to consider a line
    lines = cv2.HoughLinesP(morphed, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    return [line_from_array(line, scale_ratio) for line in lines], scale_ratio


def extract_lines(page: pymupdf.Page, line_detection_params: dict) -> list[Line]:
    """Extract lines from a pdf page.

    Args:
        page (pymupdf.Page): The page to extract lines from.
        line_detection_params (dict): The parameters for the line detection algorithm.

    Returns:
        list[Line]: The detected lines as a list.
    """
    lines = detect_lines_lsd(
        page,
        lsd_params=line_detection_params["lsd"],
        scale_factor=line_detection_params["pdf_scale_factor"],
    )
    lines, scale_ratio = detect_lines_hough(
        page,
        hough_params=line_detection_params["hough"],
    )

    merging_params = line_detection_params["line_merging_params"]

    # other_merge = advanced_merge_lines(
    #     lines,
    # )

    # return lines
    tol = merging_params["merging_tolerance"] * scale_ratio
    angle_tol = merging_params["angle_threshold"]
    merged = merge_parallel_lines_quadtree(lines, tol=tol, angle_threshold=angle_tol)
    return merged


# def fancy(tresh):
#     # closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
#     canny_edges = cv2.Canny(tresh, 50, 150, apertureSize=3)
#     cv2.imwrite("data/debug_cv_edge_canny.png", canny_edges)
#     kernel = np.ones((10, 10), np.uint8)
#     morphed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
#     morphed = cv2.morphologyEx(morphed, cv2.MORPH_DILATE, kernel)
#     cv2.imwrite("data/debug_cv_edge_morphed.png", morphed)

#     # Step 3: Line segment detection using Probabilistic Hough Transform
#     min_line_length = 300  # Minimum length of line (adjust based on your image)
#     max_line_gap = 5  # Maximum allowed gap between line segments
#     rho = 10  # Distance resolution in pixels
#     theta = np.pi / 45  # Angular resolution in radians pi/60 = possible angles every 4 degree
#     threshold = 10000  # Minimum number of votes to consider a line

#     lines2 = cv2.HoughLinesP(morphed, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
#     return lines2


# def fancy(gray):
#     # First approach: Simple Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Second approach: CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_image = clahe.apply(gray)

#     # Apply adaptive thresholding to both approaches
#     thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#     thresh2 = cv2.adaptiveThreshold(clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#     # Combine results
#     combined_thresh = cv2.bitwise_or(thresh1, thresh2)
#     cv2.imwrite("data/combined_tresh.png", combined_thresh)

#     # Step 2: Multi-scale edge detection
#     # Use a combination of edge detection methods
#     edges1 = cv2.Canny(gray, 30, 150, apertureSize=3)
#     cv2.imwrite("data/debug_canny.png", edges1)
#     edges2 = cv2.Canny(clahe_image, 30, 150, apertureSize=3)
#     cv2.imwrite("data/debug_cann_clahe.png", edges2)

#     # Combine edges
#     combined_edges = cv2.bitwise_or(edges1, edges2)

#     # Apply morphological operations to ensure line continuity
#     kernel_line = np.ones((1, 5), np.uint8)  # Horizontal kernel
#     dilated_h = cv2.dilate(combined_edges, kernel_line, iterations=1)
#     kernel_line = np.ones((5, 1), np.uint8)  # Vertical kernel
#     dilated_v = cv2.dilate(combined_edges, kernel_line, iterations=1)

#     # Combine horizontal and vertical dilations
#     dilated_edges = cv2.bitwise_or(dilated_h, dilated_v)

#     # Additional closing operation to ensure connectivity
#     kernel = np.ones((3, 3), np.uint8)
#     closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
#     cv2.imwrite("data/debug_close_edges.png", closed_edges)

#     # Step 3: Multi-scale line detection
#     # Detect lines at multiple scales and parameters

#     # First pass - detect longer lines
#     lines1 = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=40, maxLineGap=20)

#     # Second pass - detect shorter, potentially discontinuous lines
#     lines2 = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=30)

#     # Combine all detected lines
#     all_lines = []

#     if lines1 is not None:
#         all_lines.extend([line[0] for line in lines1])

#     if lines2 is not None:
#         all_lines.extend([line[0] for line in lines2])

#     return lines1
