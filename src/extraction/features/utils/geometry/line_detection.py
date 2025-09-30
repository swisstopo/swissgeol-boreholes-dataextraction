"""Script for line detection in pdf pages."""

import logging
import os

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv
from numpy.typing import ArrayLike

from extraction.features.utils.geometry.geometric_line_utilities import merge_parallel_lines_quadtree
from utils.file_utils import read_params

from .geometry_dataclasses import Circle, Line
from .util import circle_from_array, line_from_array

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

    if lines is None:
        return []

    return [line_from_array(line, scale_factor) for line in lines]


def detect_lines_hough(page: pymupdf.Page, hough_params: dict) -> ArrayLike:
    """Given document, detect lines in the pdf using the hough transform algorithm.

    Args:
        page (pymupdf.Page): The page to detect lines in.
        hough_params (dict): The parameters for the bluring and hough transform.

    Returns:
        list[Line]: The lines detected in the pdf
    """
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    target_width = hough_params["target_width"]
    h, w = gray.shape
    scale_ratio = target_width / w  # resize to a fixed size, to work with the chosen parameters
    new_size = (target_width, int(h * scale_ratio))  # carefull size is (w,h) for resizing
    gray = cv2.resize(gray, new_size)
    os.makedirs("data/debug_cv", exist_ok=True)
    cv2.imwrite("data/debug_cv/debug_cv1_gray.png", gray)

    blur_dist = hough_params["blur_dist"]
    blurred = cv2.bilateralFilter(gray, blur_dist, 75, blur_dist * 5)  # dist, sigma_color, sigma space
    cv2.imwrite("data/debug_cv/debug_cv2_blur2.png", blurred)

    # Apply binary thresholding with otsu method, that minimizes the intra-class variance.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("data/debug_cv/debug_cv3_otsu.png", thresh)

    morph_kernel_size = hough_params["morph_kernel_size"]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, 3))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, morph_kernel_size))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    combined = cv2.bitwise_or(horizontal, vertical)
    cv2.imwrite("data/debug_cv/debug_cv4_morph_VH.png", combined)

    min_line_length = hough_params["min_line_length"]  # Minimum length of line (adjust based on your image)
    max_line_gap = hough_params["max_line_gap"]  # Maximum allowed gap between line segments
    rho = hough_params["rho"]  # Distance resolution in the accumulartor space in pixels
    theta = np.pi / 180 * hough_params["theta_deg"]  # Angular resolution in the accumulator space in radian
    threshold = hough_params["threshold"]  # Minimum number of votes to consider a line
    lines = cv2.HoughLinesP(combined, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    return [line_from_array(line, scale_ratio) for line in lines]


def detect_circles_hough(page: pymupdf.Page, hough_circles_params: dict) -> list[Circle]:
    """Detect circles in a pdf page using HoughCircles algorithm.

    Args:
        page (pymupdf.Page): The page to detect circles in.
        hough_circles_params (dict): Parameters for HoughCircles detection.

    Returns:
        list[Circle]: The detected circles as a list.
    """
    # Convert PDF page to image
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.Canny(gray, 50, 150)

    circles = cv2.HoughCircles(
        gray,
        method=cv2.HOUGH_GRADIENT,
        dp=hough_circles_params["dp"],
        minDist=hough_circles_params["min_dist"],
        param1=hough_circles_params["param1"],
        param2=hough_circles_params["param2"],
        minRadius=hough_circles_params["min_radius"],
        maxRadius=hough_circles_params["max_radius"]
    )

    if circles is None:
        return []

    # Convert detected circles to Circle objects
    circles = np.round(circles[0, :]).astype("int")
    detected_circles = []

    for (x, y, radius) in circles:
        # Convert back from scaled coordinates to PDF coordinates
        detected_circles.append(circle_from_array([x, y, radius], scale_factor=2))

    return detected_circles


def _circle_intersects_with_text(circle: Circle, text_lines: list, text_proximity_threshold: float = 2.0) -> bool:
    """Check if a circle intersects with or is too close to any text line.

    Args:
        circle (Circle): The circle to check.
        text_lines (list): List of TextLine objects.
        text_proximity_threshold (float): Minimum distance from text line edges in pixels.

    Returns:
        bool: True if the circle intersects or is too close to text, False otherwise.
    """
    for text_line in text_lines:
        # Create an expanded rectangle around the text line to account for proximity threshold
        expanded_rect = pymupdf.Rect(
            text_line.rect.x0 - text_proximity_threshold,
            text_line.rect.y0 - text_proximity_threshold,
            text_line.rect.x1 + text_proximity_threshold,
            text_line.rect.y1 + text_proximity_threshold
        )

        # Check if the circle center is within the expanded text rectangle
        if expanded_rect.contains(pymupdf.Point(circle.center.x, circle.center.y)):
            return True

        # Check if any part of the circle intersects with the expanded text rectangle
        # Find the closest point on the rectangle to the circle center
        closest_x = max(expanded_rect.x0, min(circle.center.x, expanded_rect.x1))
        closest_y = max(expanded_rect.y0, min(circle.center.y, expanded_rect.y1))

        # Calculate distance from circle center to closest point on rectangle
        distance = ((circle.center.x - closest_x) ** 2 + (circle.center.y - closest_y) ** 2) ** 0.5

        # If the distance is less than the circle radius, there's an intersection
        if distance < circle.radius:
            return True

    return False


def extract_circles(page: pymupdf.Page, line_detection_params: dict, text_lines: list = None) -> list[Circle]:
    """Extract circles from a pdf page.

    Args:
        page (pymupdf.Page): The page to extract circles from.
        line_detection_params (dict): The parameters for the circle detection algorithm.
        text_lines (list): List of TextLine objects for filtering circles on text (optional).

    Returns:
        list[Circle]: The detected circles as a list.
    """
    if "hough_circles" not in line_detection_params:
        logger.warning("No hough_circles parameters found in configuration. Returning empty circle list.")
        return []

    circles = detect_circles_hough(page, hough_circles_params=line_detection_params["hough_circles"])

    if not circles:
        return []

    # If text lines are provided, filter out circles that intersect with text
    if text_lines is not None:
        text_filtered_circles = []
        text_proximity_threshold = line_detection_params.get("hough_circles", {}).get("text_proximity_threshold", 2.0)

        for circle in circles:
            is_on_text = _circle_intersects_with_text(circle, text_lines, text_proximity_threshold)
            if not is_on_text:
                text_filtered_circles.append(circle)

        filtered_circles = text_filtered_circles
        if mlflow_tracking:
            logger.info(f"Filtered out {len(circles) - len(filtered_circles)} circles intersecting with text")
    else:
        filtered_circles = circles

    if mlflow_tracking and filtered_circles:
        logger.info(f"Detected {len(filtered_circles)} circles on page")

    return filtered_circles


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
    # We currently use lsd as it is more precise detecting smalls lines. The following comments shows how the hough
    # transform line segment detector could be called, in case we need it later on.
    #
    # lines = detect_lines_hough(page, hough_params=line_detection_params["hough"])

    if not lines:
        return []

    merging_params = line_detection_params["line_merging_params"]

    width = page.rect[2]
    scaling_factor = width / merging_params["reference_size"]

    # return lines
    tol = merging_params["merging_tolerance"] * scaling_factor
    angle_tol = merging_params["angle_threshold"]
    min_line_length = merging_params["min_line_length"] * scaling_factor
    horizontal_slope_tolerance = merging_params["horizontal_slope_tolerance"]
    merged = merge_parallel_lines_quadtree(lines, tol=tol, angle_threshold=angle_tol)
    merged = [
        line for line in merged if line.length >= min_line_length or line.is_horizontal(horizontal_slope_tolerance)
    ]
    return merged
