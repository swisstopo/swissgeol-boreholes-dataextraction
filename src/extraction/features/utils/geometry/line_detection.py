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


def extract_lines(page: pymupdf.Page, line_detection_params: dict) -> list[Line]:
    """Extract lines from a pdf page.

    Args:
        page (pymupdf.Page): The page to extract lines from.
        line_detection_params (dict): The parameters for the line detection algorithm.

    Returns:
        tuple(list[Line], list[Line]): The detected lines as a list. First the longest, then all. TODO
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
    filtered = [
        line for line in merged if line.length >= min_line_length or line.is_horizontal(horizontal_slope_tolerance)
    ]
    write_img_debug("debug.png", page, line_detection_params["pdf_scale_factor"], merged)
    return filtered, merged


def find_diags_ending_in_zone(
    lines: list[Line], line_search_zone: pymupdf.Rect, min_vertical_dist: float, max_horizontal_dist: float
):
    """Find diagonal lines ending in a given zone (note: the end of the line is always on the right).

    Args:
        lines (list[Line]): The lines to search in.
        line_search_zone (pymupdf.Rect): The zone to search for line endings in.
        min_vertical_dist (float): The minimum vertical distance a line must have to be considered.
        max_horizontal_dist (float): The maximum horizontal distance a line can have to be considered.

    Returns:
        list[Line]: The lines ending in the given zone.
    """
    angle_threshold = line_detection_params["line_merging_params"]["angle_threshold"]
    return [
        g_line
        for g_line in lines
        if line_search_zone.contains(g_line.end.tuple)
        and not g_line.is_vertical(angle_threshold)  # too many other lines are vertical
        and abs(g_line.start.y - g_line.end.y) > min_vertical_dist
        and 0 < g_line.end.x - g_line.start.x < max_horizontal_dist
    ]


def write_img_debug(path, page, pdf_scale_factor, lines):
    """Write a debug image with the detected lines drawn on it."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(pdf_scale_factor, pdf_scale_factor))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n).copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if pix.n == 4 else img

    for line in lines:
        x1, y1, x2, y2 = (line.asarray() * pdf_scale_factor).astype(int)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(path, img)
