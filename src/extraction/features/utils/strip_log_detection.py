"""Strip log structure detection utilities.

This module provides:
- Candidate detection of strip log/table-like vertical sections on a PDF page.
- Scoring (crowding, text overlap, line/circle patterns).
- Merging vertically aligned sections into full strip logs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from extraction.features.utils.text.textline import TextLine

logger = logging.getLogger(__name__)


@dataclass
class StripLogSection:
    """Single strip log section (typically one “cell block” detected)."""

    bbox: pymupdf.Rect
    confidence: float


@dataclass
class StripLog:
    """A full strip log composed of vertically aligned sections."""

    bbox: pymupdf.Rect
    sections: list[StripLogSection]

    def add_section(self, section: StripLogSection) -> None:
        """Append a section and expand this strip-log bounding box.

        Args:
            section (StripLogSection): Section to attach to this strip-log.
        """
        self.sections.append(section)
        self.bbox = self.bbox | section.bbox

    def is_aligned(self, section: StripLogSection, r_tol: float = 1e-1, a_tol: float = 5) -> bool:
        """Check if a section is vertically aligned and contiguous with this strip log.

        Args:
            section (StripLogSection): Candidate section.
            r_tol (float): Relative tolerance (0–1) for width and x0 alignment. TODO
            a_tol (float): Absolute tolerance (in page units, points) for vertical gap. TODO

        Returns:
            bool: True if the section plausibly continues this strip log.
        """
        width_err = (self.bbox.width - section.bbox.width) / self.bbox.width
        hori_err = np.abs(self.bbox.x0 - section.bbox.x0) / self.bbox.width
        # Positive if there is a gap; negative when rectangles overlap vertically.
        vert_gap = max(section.bbox.y0 - self.bbox.y1, self.bbox.y0 - section.bbox.y1)
        return (width_err < r_tol) and (hori_err < r_tol) and (vert_gap < a_tol)

    @classmethod
    def from_striplog_sections(cls, sections: list[StripLogSection] | StripLogSection) -> StripLog:
        """Create a StripLog from one or many sections.

        Args:
            sections (list[StripLogSection] | StripLogSection): One section or a list of sections.

        Returns:
            StripLog: A StripLog whose bbox is the union of the provided sections.
        """
        if isinstance(sections, StripLogSection):
            sections = [sections]

        # Get bounding boxes
        bbox: pymupdf.Rect = None
        for section in sections:
            bbox = bbox | section.bbox if bbox else section.bbox

        return cls(bbox, sections)


def _page_to_grayscale(page: pymupdf.Page, dpi: int = 72) -> tuple[np.ndarray | None, float]:
    """Render a page to a grayscale numpy array.

    Args:
        page (pymupdf.Page): PyMuPDF page.
        dpi (int): Rendering resolution. Defaults to 72.

    Returns:
        np.ndarray | None: An array whose shape is defined by dpi.
        float: Scaling factor to page units per pixel
    """
    # Get area based on clip and dpi
    pix_hd = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    dpi_scaling = page.rect.height / pix_hd.h

    # Read image from buffer
    gray = np.frombuffer(pix_hd.samples, dtype=np.uint8).reshape(pix_hd.h, pix_hd.w)
    return gray, dpi_scaling


def _threshold_image(gray: np.ndarray, block_size: int, c: int, kernel_size: int = 5) -> np.ndarray:
    """Adaptive thresholding (robust to uniform backgrounds and small noise).

    Args:
        gray (np.ndarray): A NxM grayscale image.
        block_size (int): Odd window size for local mean.
        c (int): Constant subtracted from local mean.
        kernel_size(int): Kernel size for image blurring. Defaults to 5.

    Returns:
        np.ndarray: Binary NxM image with values {0, 1}.
    """
    # Apply Gaussian blur to smooth the image and reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Thresholding: Use adaptive for areas that ar enot white but filled with a unique color
    thr = cv2.adaptiveThreshold(gray_blurred, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    return thr


def _detect_candidates_from_page(
    page: pymupdf.Page,
    dpi_params: dict,
    threshold_param: dict,
    table_param: dict,
) -> list[pymupdf.Rect]:
    """Detect table cell–like candidates on a PDF page via morphological line extraction.

    Args:
        page (pymupdf.Page): The page to process.
        dpi_params (dict): Rendering parameters.
        threshold_param (dict): Thresholding parameters.
        table_param (dict): Table detection geometric/morphological constraints

    Returns:
        list[pymupdf.Rect]: List of candidate rectangles in page coordinates.
    """
    # Render
    image, rescale_factor = _page_to_grayscale(page=page, dpi=dpi_params["table"])

    # Threshold
    thr = _threshold_image(image, **threshold_param)

    # Line extraction
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (table_param["min_line_length"], 1))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, table_param["min_line_length"]))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_ker, iterations=1)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_ker, iterations=1)
    horiz_closed = cv2.morphologyEx(horiz, cv2.MORPH_DILATE, h_ker)
    vert_closed = cv2.morphologyEx(vert, cv2.MORPH_DILATE, v_ker)

    # Merge and thicken to close tiny gaps
    grid = (cv2.add(horiz_closed, vert_closed) > 0).astype(np.uint8)
    grid = cv2.dilate(
        grid, np.ones((table_param["min_line_dilation"], table_param["min_line_dilation"]), np.uint8), iterations=1
    )

    # Remove small connected components (in the line mask)
    label_img = label(grid)
    label_img = remove_small_objects(
        label_img, min_size=4 * table_param["min_line_length"] * table_param["min_line_dilation"]
    )

    # Candidates are large background “cells”
    label_img_inv = label(label_img == 0)

    # Region filtering and bbox extraction
    candidates: list[pymupdf.Rect] = []
    for region in regionprops(label_img_inv):
        if (
            (region.image.shape[1] < table_param["min_cell_width"])
            or (region.image.shape[1] > table_param["max_cell_width"])
            or (region.image.shape[0] < table_param["min_cell_height"])
            or (region.area / region.area_bbox < table_param["min_area_ratio"])
        ):
            continue
        # Invert coordinate system (x;y) and (y;x) due to packages inconsitency
        candidates.append(pymupdf.Rect(region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]))

    # Convert px -> page coords
    candidates = _rescale_bboxes(candidates, scale=rescale_factor)

    return candidates


def _rescale_bboxes(bboxes: list[pymupdf.Rect], scale: float) -> list[pymupdf.Rect]:
    """Adapt detected striplog area to match project rendering at base dpi.

    Args:
        bboxes (list[pymupdf.Rect]): Bounding boxres to rescale.
        scale (float): Scaling factor for bounding boxes.

    Returns:
        list[pymupdf.Rect]: Rescaled bbounding boxes.
    """
    return [pymupdf.Rect((scale * np.array(bbox)).astype(int).tolist()) for bbox in bboxes]


def _score_crowding(image: np.ndarray, kernel_size: int = 5) -> float:
    """Return foreground density after dilation (range [0, 1]).

    Args:
        image (np.ndarray): Grayscale patch (uint8).
        kernel_size (int): Square kernel size for dilation. Defaults to 5.

    Returns:
        float: Density in [0, 1] computed as the mean of the dilated binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=1)
    return closed.mean()


def _score_text_disjoint(rect: pymupdf.Rect, text_lines: list[TextLine]) -> float:
    """Returns fraction of area not obstructed by text.

    Args:
        rect (pymupdf.Rect): Region of interest in page coordinates.
        text_lines (list[TextLine]): Optional list of detected text lines in page coordinates.

    Returns:
        (float) Score in [0, 1]; 1.0 means no text overlap, 0.0 means fully covered by text.
    """
    if not text_lines:
        return 1.0
    # Detect all text overlapping area with region and sums areas
    text_within = [line for line in text_lines if rect.intersects(line.rect)]
    # Remove text that was produce by ,aterial structure (looking like 0 or 8)
    text_within = [line for line in text_within if not _is_numeric_pattern(line.text)]
    # Extract areas and sum
    text_area_within = np.sum([(line.rect & rect).get_area() for line in text_within])
    # Text penalty given as area occupied by text
    score = 1 - text_area_within / rect.get_area()
    return score


def _score_line(image: np.ndarray, lsd_params: dict) -> float:
    """Estimate line “area” fraction via LSD.

    Args:
        image (np.ndarray): Grayscale patch.
        lsd_params (dict): Keyword args for `cv2.createLineSegmentDetector`.

    Returns:
        float: Fraction of area covered by lines.
    """
    # Create default line segment detector
    #  Documentation for the parameters can be found here:
    #  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gae0bba3b867a5f44d1b823aef4f57ee8d
    lsd = cv2.createLineSegmentDetector(**lsd_params)

    # Detect lines in the image
    lines, lines_width, _, _ = lsd.detect(image)

    if lines is None:
        return 0.0

    # Compute line length
    lines_length = np.sqrt((lines[:, 0, 0] - lines[:, 0, 2]) ** 2 + (lines[:, 0, 1] - lines[:, 0, 3]) ** 2)
    lines_area = np.sum(lines_width.flatten() * lines_length)

    return lines_area / float(image.size)
    # return [line_from_array(line, scale_factor) for line in lines]


def _score_circle(image: np.ndarray, hough_circles_params: dict) -> float:
    """Estimate circle area fraction via HoughCircles (0..1).

    Ref: https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html

    Args:
        image (np.ndarray): Grayscale patch (single channel).
        hough_circles_params (dict): Dict with keys:
            - dp: Inverse ratio of the accumulator resolution to the image resolution.
            - min_dist: Minimum distance between the centers of the detected circles.
            - param1: Higher threshold for the Canny edge detector (the lower one is twice smaller).
            - param2: Accumulator threshold for the circle centers at the detection stage.
            - min_radius: Minimum circle radius.
            - max_radius: Maximum circle radius.

    Returns:
        float: Fraction of area covered by circles.
    """
    # Apply Gaussian blur to smooth the image and reduce noise
    image = cv2.GaussianBlur(image, (hough_circles_params["kernel_size"], hough_circles_params["kernel_size"]), 0)

    # Detect circles
    circle_arrays = cv2.HoughCircles(
        image,
        method=cv2.HOUGH_GRADIENT,
        dp=hough_circles_params["dp"],
        minDist=hough_circles_params["min_dist"],
        param1=hough_circles_params["param1"],
        param2=hough_circles_params["param2"],
        minRadius=hough_circles_params["min_radius"],
        maxRadius=hough_circles_params["max_radius"],
    )

    if not circle_arrays:
        return 0

    # Convert detected circles to Circle objects
    circle_area = np.sum(np.pi * circle_arrays[0, :, 2] ** 2)
    return circle_area / image.size


def _score_striplogs(
    bboxes: list[pymupdf.Rect],
    page,
    dpi_params: dict,
    threshold_param: dict,
    hough_circles_params: dict,
    lsd_params: dict,
    score_params: dict,
    text_lines: list[TextLine] = None,
) -> list[StripLogSection]:
    """Score candidate strip-log regions by visual/text features and return sections.

    Args:
        bboxes (list[pymupdf.Rect]): Candidate regions to be scored.
        page (_type_): The source page.
        dpi_params (dict): Rendering parameters.
        threshold_param (dict): Thresholding parameters.
        hough_circles_params (dict): Parameters forwarded to OpenCV HoughCircles.
        lsd_params (dict): Parameters forwarded to OpenCV LSD.
        score_params (dict): High-level scoring toggles/weights.
        text_lines (list[TextLine], optional): Detected text lines in page. Defaults to None.

    Returns:
        list[StripLogSection]: Scored sections with confidence score.
    """
    image, scaling_factor = _page_to_grayscale(page=page, dpi=dpi_params["score"])
    image = _threshold_image(image, **threshold_param)

    strip_candidates = []
    bboxes_px = _rescale_bboxes(bboxes, scale=1 / scaling_factor)

    for bbox, bbox_px in zip(bboxes, bboxes_px, strict=True):
        # 1) Get scaled area
        bbox_image = image[int(bbox_px.y0) : int(bbox_px.y1), int(bbox_px.x0) : int(bbox_px.x1)]

        # 2) Compute local stats
        confidence = 1.0

        if score_params["crowding"]:
            confidence *= _score_crowding(bbox_image)
        if score_params["text_overlap"]:
            confidence *= _score_text_disjoint(bbox, text_lines=text_lines)
        if score_params["pattern"]:
            circle_score = _score_circle(bbox_image, hough_circles_params)
            line_score = _score_line(bbox_image, lsd_params)
            confidence *= np.clip(circle_score + 4 * line_score, a_min=0, a_max=1)

        strip_candidates.append(
            StripLogSection(
                bbox=bbox,
                confidence=confidence,
            )
        )
    return strip_candidates


def _merge_sections(sections: list[StripLogSection], min_sections: int) -> list[StripLog]:
    """Merge vertically aligned sections into strip-logs.

    Args:
        sections (list[StripLogSection]): Detected section candidates.
        min_sections (int): Minimum number of sections required for a StripLog to be kept.

    Returns:
        list[StripLog]: A list of merged `StripLog` objects.
    """
    # Check if at leat one section detected
    if not sections:
        return []

    # Sort bounding boxes top to botttom
    sections.sort(key=lambda section: section.bbox[1])

    # Merge strip logs
    striplogs: list[StripLog] = [StripLog.from_striplog_sections(sections=sections[0])]
    for section in sections[1:]:
        # Check if can be added
        is_aligned = [striplog.is_aligned(section) for striplog in striplogs]
        if any(is_aligned):
            # If multiple matches, take first
            index = np.argmax(is_aligned)
            striplogs[index].add_section(section)
        else:
            striplogs.append(StripLog.from_striplog_sections(sections=section))

    # Filterout strips where not enough elements
    return [striplog for striplog in striplogs if len(striplog.sections) >= min_sections]


def _is_numeric_pattern(text: str) -> bool:
    """Pattern for combinations of digits and dots (0, 0.0, 00, 0.0.0, 08, .00, 0-8 etc.).

    Args:
        text (str): Raw text to classify.

    Returns:
        bool: True if text is composed artifact characters.
    """
    numeric_pattern = re.compile(r"^[\s08./-]+$")
    return bool(numeric_pattern.match(text))


def detect_strip_logs(
    page: pymupdf.Page,
    text_lines: list[TextLine],
    striplog_detection_params: dict,
) -> list[StripLog]:
    """End-to-end strip-log detection and merging.

    Args:
        page (pymupdf.Page): Page to process.
        text_lines (list[TextLine]): Detected text lines in page space.
        striplog_detection_params (dict): Nested config with groups.

    Returns:
        list[StripLog]: List of merged StripLog objects that passed the confidence threshold.
    """
    # Extract parameter groups
    dpi_params = striplog_detection_params.get("dpi", {})
    threshold_param = striplog_detection_params.get("threshold", {})
    table_param = striplog_detection_params.get("table", {})
    hough_circles_params = striplog_detection_params.get("hough_circles", {})
    lsd_params = striplog_detection_params.get("lsd", {})
    score_params = striplog_detection_params.get("score", {})

    # Candidate detection in page space
    bboxes = _detect_candidates_from_page(page, dpi_params, threshold_param, table_param)

    # Candidate scoring
    section_candidates = _score_striplogs(
        bboxes, page, dpi_params, threshold_param, hough_circles_params, lsd_params, score_params, text_lines
    )

    # Confidence filtering
    section_filtered = [
        strip for strip in section_candidates if strip.confidence >= striplog_detection_params["confidence"]
    ]

    # Vertical merging, then filter by minimum section count.
    return _merge_sections(section_filtered, min_sections=striplog_detection_params["min_sections"])
