"""Script for line detection in pdf pages."""

import logging

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv

from utils.file_utils import read_params

import rtree
from .geometry_dataclasses import Circle

load_dotenv()

logger = logging.getLogger(__name__)

line_detection_params = read_params("line_detection_params.yml")


def detect_circles_hough(page: pymupdf.Page, hough_circles_params: dict) -> list[Circle]:
    """Detect circles in a pdf page using HoughCircles algorithm.

    For more infromation around the parameters see: https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html

    Args:
        page (pymupdf.Page): The page to detect circles in.
        hough_circles_params (dict): Parameters for HoughCircles detection.
            - dp: Inverse ratio of the accumulator resolution to the image resolution.
            - min_dist: Minimum distance between the centers of the detected circles.
            - param1: Higher threshold for the Canny edge detector (the lower one is twice smaller).
            - param2: Accumulator threshold for the circle centers at the detection stage.
            - min_radius: Minimum circle radius.
            - max_radius: Maximum circle radius.

    Returns:
        list[Circle]: The detected circles as a list.
    """
    # Convert PDF page to image
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

    # Apply Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circle_arrays = cv2.HoughCircles(
        gray,
        method=cv2.HOUGH_GRADIENT,
        dp=hough_circles_params["dp"],
        minDist=hough_circles_params["min_dist"],
        param1=hough_circles_params["param1"],
        param2=hough_circles_params["param2"],
        minRadius=hough_circles_params["min_radius"],
        maxRadius=hough_circles_params["max_radius"],
    )

    if circle_arrays is None:
        return []

    # Convert detected circles to Circle objects
    circle_arrays = np.round(circle_arrays[0, :]).astype("int")

    return [Circle.circle_from_array(array, scale_factor=1) for array in circle_arrays]

def _build_text_rtree(text_lines: list) -> rtree.index.Index:
    """Build spatial index for text lines.

    Args:
        text_lines (list): List of TextLine objects.
    Returns:
        rtree.index.Index: RTree spatial index of text lines.
    """
    text_rtree = rtree.index.Index()
    for line in text_lines:
        # Insert text rectangle into RTree
        text_rtree.insert(
            id(line),
            (line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1),  # Bounding box
            obj=line  # Store the actual object for later use
        )
    return text_rtree

def _circle_intersects_text_rtree(
    circle: Circle,
    text_rtree: rtree.index.Index,
    text_proximity_threshold: float
) -> bool:
    """Check if circle intersects with text using spatial index.

    Using RTree spatial index to efficiently query text lines near the circle.

    Args:
        circle (Circle): The circle to check.
        text_rtree (rtree.index.Index): RTree spatial index of text lines.
        text_proximity_threshold (float): The distance threshold to consider proximity to text.
    Returns:
        bool: True if the circle intersects with or is too close to any text line, False
    """
    # Define query bounding box around circle (with threshold)
    query_bbox = (
        circle.center.x - circle.radius - text_proximity_threshold,  # x0
        circle.center.y - circle.radius - text_proximity_threshold,  # y0
        circle.center.x + circle.radius + text_proximity_threshold,  # x1
        circle.center.y + circle.radius + text_proximity_threshold   # y1
    )

    # Query RTree for potentially intersecting text lines
    #candidate_lines = list(text_rtree.intersection(query_bbox, objects="raw"))

    # Check actual intersection only for candidates
    for text_line in text_rtree.intersection(query_bbox, objects="raw"):

        # Find closest point on rectangle to circle center
        closest_x = max(text_line.rect.x0, min(circle.center.x, text_line.rect.x1))
        closest_y = max(text_line.rect.y0, min(circle.center.y, text_line.rect.y1))

        # Calculate distance
        distance = ((circle.center.x - closest_x) ** 2 +
                   (circle.center.y - closest_y) ** 2) ** 0.5

        # Check intersection
        if distance < circle.radius + text_proximity_threshold:
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
    hough_circle_parameters = line_detection_params["hough_circles"]

    circles = detect_circles_hough(page, hough_circles_params=hough_circle_parameters)

    if not circles:
        return []

    # If text lines are provided, filter out circles that intersect with text
    if text_lines is not None:
        text_proximity_threshold = hough_circle_parameters.get("text_proximity_threshold")

        text_rtree = _build_text_rtree(text_lines)

        filtered_circles = [
            circle
            for circle in circles
            if not _circle_intersects_text_rtree(circle, text_rtree, text_proximity_threshold)
        ]

    else:
        filtered_circles = circles

    return filtered_circles
