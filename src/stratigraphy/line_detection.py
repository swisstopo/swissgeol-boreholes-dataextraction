"""Script for line detection in pdf pages."""

import logging
import os

import cv2
import fitz
import numpy as np
from dotenv import load_dotenv
from numpy.typing import ArrayLike

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.geometric_line_utilities import (
    drop_vertical_lines,
    merge_parallel_lines_quadtree,
)
from stratigraphy.util.util import line_from_array, read_params

load_dotenv()

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

line_detection_params = read_params("line_detection_params.yml")


def detect_lines_lsd(page: fitz.Page, scale_factor=2, lsd_params=None) -> ArrayLike:
    """Given a file path, detect lines in the pdf using the Line Segment Detector (LSD) algorithm.

    Publication of the algorithm can be found here: http://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf
    Note: As of now the function only works for pdfs with a single page.
          For now the function displays each pdf with the lines detected using opencv.
          This behavior will be changed in the future.

    Args:
        page (fitz.Page): The page to detect lines in.
        scale_factor (float, optional): The scale factor to scale the pdf page. Defaults to 2.
        lsd_params (dict, optional): The parameters for the Line Segment Detector. Defaults to None.

    Returns:
        list[Line]: The lines detected in the pdf.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create default line segment detector
    lsd = cv2.createLineSegmentDetector(**lsd_params)
    #  Documentation for the parameters can be found here:
    #  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gae0bba3b867a5f44d1b823aef4f57ee8d

    # Detect lines in the image
    lines = lsd.detect(gray)[0]
    return [line_from_array(line, scale_factor) for line in lines]


def extract_lines(page: fitz.Page, line_detection_params: dict) -> list[Line]:
    """Extract lines from a pdf page.

    Args:
        page (fitz.Page): The page to extract lines from.
        line_detection_params (dict): The parameters for the line detection algorithm.

    Returns:
        list[Line]: The detected lines as a list.
    """
    lines = detect_lines_lsd(
        page,
        lsd_params=line_detection_params["lsd"],
        scale_factor=line_detection_params["pdf_scale_factor"],
    )
    lines = drop_vertical_lines(lines, threshold=line_detection_params["vertical_lines_threshold"])
    merging_params = line_detection_params["line_merging_params"]

    return merge_parallel_lines_quadtree(
        lines, tol=merging_params["merging_tolerance"], angle_threshold=merging_params["angle_threshold"]
    )
