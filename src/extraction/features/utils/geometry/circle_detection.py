"""Script for line detection in pdf pages."""

import logging
import os

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv

from utils.file_utils import read_params

from .geometry_dataclasses import Circle
from .util import circle_from_array

load_dotenv()

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

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
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

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
    """Check if a circle intersects with or is too close to any text line."""
    for text_line in text_lines:
        # Create an expanded rectangle around the text line
        expanded_rect = pymupdf.Rect(
            text_line.rect.x0 - text_proximity_threshold,
            text_line.rect.y0 - text_proximity_threshold,
            text_line.rect.x1 + text_proximity_threshold,
            text_line.rect.y1 + text_proximity_threshold
        )

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
    hough_circle_parameters = line_detection_params["hough_circles"]

    circles = detect_circles_hough(page, hough_circles_params=hough_circle_parameters)

    if not circles:
        return []

    # If text lines are provided, filter out circles that intersect with text
    if text_lines is not None:
        text_proximity_threshold = hough_circle_parameters.get("text_proximity_threshold")

        filtered_circles = [
            circle for circle in circles
            if not _circle_intersects_with_text(circle, text_lines, text_proximity_threshold)
        ]

    else:
        filtered_circles = circles

    return filtered_circles