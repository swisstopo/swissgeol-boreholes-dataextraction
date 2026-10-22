"""Script for line detection in pdf pages."""

import logging
import statistics
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv

from swissgeol_doc_processing.geometry.geometric_line_utilities import merge_parallel_lines_quadtree
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.geometry.util import line_from_array
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_rtree import TextLineRTree

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LineDetectionResult:
    """Class containing the results of the geometric line detection."""

    long_or_horizontal_lines: list[Line]
    all_geometric_lines: list[Line]
    cv_scale_factor: float


def detect_lines_fld(page: pymupdf.Page, scale_factor=2, fld_params=None) -> list[Line]:
    """Given a PDF page, detect lines using the Fast Line Detector (FLD) algorithm.

    Publication of the algorithm can be found here: https://www.researchgate.net/publication/283992330

    Args:
        page (pymupdf.Page): The page to detect lines in.
        scale_factor (float, optional): The scale factor to scale the PDF page. Defaults to 2.
        fld_params (dict, optional): The parameters for the Fast Line Detector. Defaults to None.

    Returns:
        list[Line]: The lines detected on the PDF page.
    """
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_factor, scale_factor), colorspace=pymupdf.csGRAY)
    reshaped = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

    # Create default line segment detector
    fld = cv2.ximgproc.createFastLineDetector(**fld_params)

    # Detect lines in the image
    lines = fld.detect(reshaped)

    if lines is None:
        return []

    return [line_from_array(line, scale_factor) for line in lines]


def extract_lines(page: pymupdf.Page, text_lines: list[TextLine], line_detection_params: dict) -> LineDetectionResult:
    """Extract lines from a pdf page.

    Geometric lines that fall entirely within a single word of some text line will be ignored.

    Args:
        page (pymupdf.Page): The page to extract lines from.
        text_lines (list[TextLine]): List of text lines.
        line_detection_params (dict): The parameters for line detection.

    Returns:
        LineDetectionResult
    """
    horizontal_text_heights = [line.rect.height for line in text_lines if abs(line.text_angle) < 5]
    if len(horizontal_text_heights) > 2:
        median_text_height = statistics.median(horizontal_text_heights)
        expected_text_height = line_detection_params["expected_text_height"]
        cv_scale_factor = expected_text_height / median_text_height
    else:
        cv_scale_factor = line_detection_params["default_pdf_scale_factor"]

    lines = detect_lines_fld(
        page,
        fld_params=line_detection_params["fld"],
        scale_factor=cv_scale_factor,
    )
    # We currently use lsd as it is more precise detecting smalls lines. The following comments shows how the hough
    # transform line segment detector could be called, in case we need it later on.
    #
    # lines = detect_lines_hough(page, hough_params=line_detection_params["hough"])

    def grow_rect(rect: pymupdf.Rect, delta: float) -> pymupdf.Rect:
        return pymupdf.Rect(rect.x0 - delta, rect.y0 - delta, rect.x1 + delta, rect.y1 + delta)

    line_rtree = TextLineRTree(text_lines)

    def line_is_valid(line: Line) -> bool:
        line_bbox = pymupdf.Rect(
            x0=min(line.start.x, line.end.x),
            y0=min(line.start.y, line.end.y),
            x1=max(line.start.x, line.end.x),
            y1=max(line.start.y, line.end.y),
        )
        matching_text_lines = line_rtree.query(line_bbox)
        # grow each word's rect by 1 point because the line detection algorithm is only accurate to +/- 1 pixel
        return all(
            line.length > word.rect.height or not grow_rect(word.rect, 1).contains(line_bbox)
            for text_line in matching_text_lines
            for word in text_line.words
        )

    lines = [line for line in lines if line_is_valid(line)]
    if not lines:
        return [], []

    merging_params = line_detection_params["line_merging_params"]

    # return lines
    tol = merging_params["merging_tolerance"]
    angle_threshold = merging_params["angle_threshold"]
    min_line_length = merging_params["min_line_length"]
    merged = merge_parallel_lines_quadtree(lines, tol=tol, angle_threshold=angle_threshold)
    filtered = [line for line in merged if line.length >= min_line_length or line.is_horizontal(angle_threshold)]
    return LineDetectionResult(
        long_or_horizontal_lines=filtered, all_geometric_lines=merged, cv_scale_factor=cv_scale_factor
    )
