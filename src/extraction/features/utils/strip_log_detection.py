"""striplog structure detection utilities.

This module provides:
- Candidate detection of striplog/table-like vertical sections on a PDF page.
- Scoring (crowding, text overlap, line/circle patterns).
- Merging vertically aligned sections into full striplogs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
from scipy.sparse.csgraph import connected_components
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.cluster import DBSCAN

from extraction.features.utils.text.textline import TextLine

logger = logging.getLogger(__name__)


@dataclass
class StripLogSection:
    """Single striplog section (typically one “cell block” detected)."""

    bbox: pymupdf.Rect
    confidence: float = 1.0

    def aligns(self, section: StripLogSection, r_tol: float = 1e-1) -> bool:
        """Check if a section is vertically aligned and contiguous with this striplog.

        Args:
            section (StripLogSection): Candidate section.
            r_tol (float): Relative tolerance (0–1) for width and x0 alignment.

        Returns:
            bool: True if the section plausibly continues this striplog.
        """
        width_err = np.abs(self.bbox.width - section.bbox.width) / self.bbox.width
        hori_err = np.abs(self.bbox.x0 - section.bbox.x0) / self.bbox.width
        return (width_err <= r_tol) and (hori_err <= r_tol)


@dataclass
class StripLog:
    """A full striplog composed of vertically aligned sections."""

    bbox: pymupdf.Rect
    sections: list[StripLogSection]

    def add_section(self, section: StripLogSection) -> None:
        """Append a section and expand this strip-log bounding box.

        Args:
            section (StripLogSection): Section to attach to this strip-log.
        """
        self.sections.append(section)
        self.bbox = self.bbox | section.bbox

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


def _page_to_grayscale(page: pymupdf.Page, dpi: int = 72, use_blur: bool = True) -> tuple[np.ndarray, float]:
    """Render a page to a grayscale numpy array.

    Args:
        page (pymupdf.Page): PyMuPDF page.
        dpi (int): Rendering resolution. Defaults to 72.
        use_blur (bool): Apply blur to output image with standard kernel.

    Returns:
        np.ndarray: An array whose shape is defined by dpi.
        float: Scaling factor to page units per pixel
    """
    # Get area based on clip and dpi
    pix_hd = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    dpi_scaling = page.rect.height / pix_hd.h

    # Read image from buffer
    gray = np.frombuffer(pix_hd.samples, dtype=np.uint8).reshape(pix_hd.h, pix_hd.w)

    # Blur is needed
    if use_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray, dpi_scaling


def _threshold_image(im_gray: np.ndarray, block_size: int, c: int, clean_size: int) -> np.ndarray:
    """Adaptive thresholding (robust to uniform backgrounds and small noise).

    Args:
        im_gray (np.ndarray): A NxM grayscale image.
        block_size (int): Odd window size for local mean.
        c (int): Constant subtracted from local mean.
        clean_size (int): Minimal area to be considered in image. Used to remove small dot artifacts.

    Returns:
        np.ndarray: Binary NxM image with values {0, 1}.
    """
    # Thresholding: adaptive to handle uniformly colored backgrounds. clean_size removes tiny speckles.
    thr = cv2.adaptiveThreshold(im_gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    thr = remove_small_objects(thr.astype(bool), min_size=clean_size).astype(np.uint8)
    return thr


def _detect_table_from_thresholded(
    im_binary: np.ndarray, min_line_length: int, min_line_dilation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect a table grid and label its cells from a binary page image.

    Given a binarized image, this routine extracts horizontal and vertical line masks via
    morphological operations. It removes tiny spurious components, then labels connected background
    regions inside the grid as candidate table cells. It also returns labeled
    masks for vertical and horizontal lines.

    Args:
        im_binary (np.ndarray): Binary image with values {0,1}.
        min_line_length (int): Minimum line length in pixels used to build the rectangular structuring
            elements for the morphological operations.
        min_line_dilation (int): Square dilation kernel size (in pixels). Increases line thickness to bridge small
            gaps and ensure grid connectivity.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - cells_labeled (np.ndarray): Labeled image of candidate table cells,
              obtained by labeling background regions within the cleaned grid {0, 1, ..., N}.
            - vert_labeled (np.ndarray): Labeled image of vertical line components
              after opening/dilation values {0, 1, ..., V}.
            - horiz_labeled (np.ndarray): Labeled image of horizontal line components
              after opening/dilation values {0, 1, ..., H}.

    """
    # Line extraction
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
    horiz = cv2.morphologyEx(im_binary, cv2.MORPH_OPEN, h_ker, iterations=1)
    vert = cv2.morphologyEx(im_binary, cv2.MORPH_OPEN, v_ker, iterations=1)

    # Only perform vertical opening to avoid connecting sections
    vert_closed = cv2.morphologyEx(vert, cv2.MORPH_DILATE, v_ker)

    # Merge and thicken to close tiny gaps
    grid = (cv2.add(horiz, vert_closed) > 0).astype(np.uint8)
    grid = cv2.dilate(grid, np.ones((min_line_dilation, min_line_dilation), np.uint8), iterations=1)

    # Remove small connected components (in the line mask)
    label_img = label(grid)
    label_img = remove_small_objects(label_img, min_size=4 * min_line_length * min_line_dilation)

    # Candidates are large background “cells”
    label_img_inv = label(label_img == 0)

    return label_img_inv, label(vert_closed), label(horiz)


def _detect_candidates_from_page(
    im_binary: np.ndarray,
    table_param: dict,
) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """Detect table cell–like candidates on a PDF page via morphological line extraction.

    Args:
        im_binary (np.ndarray): Binary image with values {0,1} for table detection.
        table_param (dict): Table detection geometric/morphological constraints

    Returns:
        list[pymupdf.Rect]: List of candidate rectangles in page coordinates.
        list[pymupdf.Rect]: List of vertical structure lines bounding boxes.
    """
    im_table_labeled, im_vert_labeled, _ = _detect_table_from_thresholded(
        im_binary, table_param["min_line_length"], table_param["min_line_dilation"]
    )

    structures_bbox: list[pymupdf.Rect] = []
    for region in regionprops(im_vert_labeled):
        # Invert coordinate system (x;y) and (y;x) due to packages inconsistency
        structures_bbox.append(
            pymupdf.Rect(
                region.bbox[1] - table_param["margin_vert_line"],
                region.bbox[0],
                region.bbox[3] + table_param["margin_vert_line"],
                region.bbox[2],
            )
        )

    # Region filtering and bbox extraction
    candidates_bbox: list[pymupdf.Rect] = []
    for region in regionprops(im_table_labeled):
        if (
            (region.image.shape[1] < table_param["min_cell_width"])
            or (region.image.shape[1] > table_param["max_cell_width"])
            or (region.image.shape[0] < table_param["min_cell_height"])
            or (region.area / region.area_bbox < table_param["min_area_ratio"])
        ):
            continue
        # Invert coordinate system (x;y) and (y;x) due to packages inconsistency
        candidates_bbox.append(pymupdf.Rect(region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]))

    return candidates_bbox, structures_bbox


def _rescale_bboxes(bboxes: list[pymupdf.Rect], scale: float) -> list[pymupdf.Rect]:
    """Rescale bounding boxes by a scalar factor to match project rendering at base dpi.

    Args:
        bboxes (list[pymupdf.Rect]): Bounding boxes to rescale.
        scale (float): Scaling factor for bounding boxes.

    Returns:
        list[pymupdf.Rect]: Rescaled bounding boxes.
    """
    return [pymupdf.Rect((scale * np.array(bbox)).astype(int).tolist()) for bbox in bboxes]


def _score_crowding(im_binary: np.ndarray, kernel_size: int = 5, margin: int = 2) -> float:
    """Return foreground density after dilation (range [0, 1]).

    This function estimates how visually "crowded" a binary image region is by
    measuring the average pixel density after applying a morphological dilation.
    The dilation expands the foreground regions using a disk-like (elliptical) kernel, which helps
    capture nearby pixel clusters and fill small gaps. A higher score indicates
    denser or more cluttered regions, while a lower score suggests sparse content.

    Args:
        im_binary (np.ndarray): Binary image where foreground pixels represent detected features.
        kernel_size (int): Circle kernel used to expand foreground regions. Defaults to 5.
        margin (int): Number of pixels to crop from each border before computation, to reduce border artifacts.

    Returns:
        float: Foreground density in [0, 1] representing the mean intensity of the dilated binary mask.
    """
    im_binary = im_binary[margin:-margin, margin:-margin]
    if len(im_binary) == 0:
        return 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(im_binary, cv2.MORPH_DILATE, kernel, iterations=1)
    return np.clip(closed.mean(), a_min=0.0, a_max=1.0)


def _is_text(
    rect: pymupdf.Rect, text_lines: list[TextLine], tau: float = 5.0, penalty: float = 1.0, threshold: float = 0.1
) -> bool:
    """Return True if the region is effectively contains text.

    The method computes a penalty score in [0, 1] that reflects how much text overlaps
    the region: higher overlap/longer text → higher penalty. The region is considered
    positive if the final penalty is lower than `threshold`.

    Scoring (two factors, then combined and exponentiated):
      - score area: proportion of the region covered by text
      - length factor: penalizes long text lines that overlap the region

    Args:
        rect (pymupdf.Rect): Region of interest in page coordinates.
        text_lines (list[TextLine]): Detected text lines in page coordinates.
        tau (float): Length scale controlling tolerance to long text (larger = more tolerant), default 5.0.
        penalty (float): Exponent controlling penalty steepness (larger = harsher), default 1.0
        threshold : Decision threshold on the final penalty. If (penalty_score >= threshold),
            the region is considered to be free of text, default 0.1.

    Returns:
        bool: True if the region is contains text (penalty < threshold), False otherwise.
    """
    if not text_lines:
        return False

    # Detect all text overlapping area with region and sums areas
    text_within = [line for line in text_lines if rect.intersects(line.rect)]
    # Remove text that was produce by OCR
    text_within_ocr = [line for line in text_within if not _is_ocr_artifact(line.text)]

    # Text penalty given as area occupied by text
    if not text_within_ocr:
        return False

    # Compute area covered by text wrt section area
    text_area_within = np.sum([(line.rect & rect).get_area() for line in text_within_ocr])

    # Compute portion of area that is overlapping with section
    score_area = 1 - text_area_within / rect.get_area()

    # measure effective length of text overlapping with section
    total_effective_length = 0
    for line in text_within_ocr:
        line_area = line.rect.get_area()
        overlap_area = (line.rect & rect).get_area()
        line_overlap_factor = overlap_area / line_area if line_area > 0 else 0
        total_effective_length += len(line.text) * line_overlap_factor

    score_length = np.exp(-total_effective_length / tau)

    return np.clip((score_area * score_length) ** penalty, a_min=0.0, a_max=1.0) < threshold


def _score_pattern(im_gray: np.ndarray, pattern_params: dict) -> float:
    """Score how strongly a page exhibits the “pattern” (lines/circles).

    The function detects straight line segments (via OpenCV’s LSD) and circular features
    (via Hough Circles), clusters their bounding-box centers with DBSCAN, and computes
    the total area of the cluster bounding boxes normalized by the page area.

    Args:
        im_gray (np.ndarray): Single-channel grayscale image (H×W).
        pattern_params (dict):  Dictionary with parameters for the two detectors and clustering.

    Returns:
        float: A value in ``[0, 1]`` proportional to how much of the page is covered by
            the union of cluster bounding boxes.

    """
    hough_circles_params = pattern_params.get("hough_circles", {})
    lsd_params = pattern_params.get("lsd", {})
    _, width = im_gray.shape

    bbox_lines = _detect_line(im_gray, lsd_params)
    bbox_circles = _detect_circle(im_gray, hough_circles_params)
    bboxes = bbox_lines + bbox_circles

    if not bboxes:
        return 0.0

    bboxes_centers = [[(x0 + x1) / 2, (y0 + y1) / 2] for x0, y0, x1, y1 in bboxes]
    cluster_labels = DBSCAN(eps=pattern_params["max_dbscan_ratio"] * width, min_samples=1).fit_predict(bboxes_centers)

    # # Infer clusters areas
    clusters_area = 0
    for cluster_id in np.unique(cluster_labels):
        x0, y0, x1, y1 = np.array(bboxes)[cluster_id == cluster_labels].T
        clusters_area += (x1.max() - x0.min()) * (y1.max() - y0.min())

    return np.clip(clusters_area / im_gray.size, a_min=0.0, a_max=1.0)


def _detect_line(im_gray: np.ndarray, lsd_params: dict) -> list[pymupdf.Rect]:
    """Detect straight line segments using OpenCV’s Line Segment Detector (LSD).

    Internally crops a uniform margin from the image to avoid boundary artifacts,
    runs LSD, and returns axis-aligned rectangles spanning each detected segment.

    Args:
        im_gray (np.ndarray): Single-channel grayscale image (H×W).
        lsd_params (dict): Parameters for LSD.

    Returns:
        list[pymupdf.Rect]: One rectangle per kept line.
    """
    # Create default line segment detector
    #  Documentation for the parameters can be found here:
    #  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gae0bba3b867a5f44d1b823aef4f57ee8d
    local_lsd_params = lsd_params.copy()
    min_line_ratio = local_lsd_params.pop("min_line_ratio")
    max_line_ratio = local_lsd_params.pop("max_line_ratio")
    max_margin = local_lsd_params.pop("max_margin")
    _, width = im_gray.shape

    # Remove small margin around to prvent detecting table lines
    im_gray_margin = im_gray[max_margin:-max_margin, max_margin:-max_margin]

    # Check if cropped area is still valid
    if im_gray_margin.size == 0:
        return []

    # Detect lines in the image
    lsd = cv2.createLineSegmentDetector(**local_lsd_params)
    lines, _, _, _ = lsd.detect(im_gray_margin)

    if lines is None:
        return []

    # Return lines that are in rang width * [min_line_length, max_line_ratio].
    return [
        pymupdf.Rect(x0, y0, x1, y1)
        for x0, y0, x1, y1 in lines[:, 0, :]
        if max_line_ratio * width > np.hypot(x1 - x0, y1 - y0) > min_line_ratio * width
    ]


def _detect_circle(im_gray: np.ndarray, hough_circles_params: dict) -> list[pymupdf.Rect]:
    """Detect circular features using Hough Circles and return bounding rectangles.

    Ref: https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html

    Args:
        im_gray (np.ndarray): Single-channel grayscale image (H×W).
        hough_circles_params (dict): Parameters forwarded to ``cv2.HoughCircles``

    Returns:
        list[pymupdf.Rect]: One bounding box per detected circle.
    """
    # Detect circles
    circle_arrays = cv2.HoughCircles(
        im_gray,
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

    return [pymupdf.Rect(x - r, y - r, x + r, y + r) for x, y, r in circle_arrays[0]]


def _score_candidates(
    bboxes: list[pymupdf.Rect],
    bboxes_dpi: list[pymupdf.Rect],
    im_gray_dpi: np.ndarray,
    im_binary_dpi: np.ndarray,
    score_params: dict,
    text_lines: list[TextLine] = None,
) -> tuple[list[float], list[bool]]:
    """Score candidate strip-log regions by visual/text features.

    Args:
        bboxes (list[pymupdf.Rect]): N candidate regions to be scored.
        bboxes_dpi (list[pymupdf.Rect]): N candidate regions to be scored at image gray and binary resolution.
        im_gray_dpi (np.ndarray): Grayscale with grayscale values {0, ..., 255}.
        im_binary_dpi (np.ndarray): Binary image with values {0,1}.
        score_params (dict): High-level scoring parameters.
        text_lines (list[TextLine], optional): Detected text lines in page. Defaults to None.

    Returns:
        list[float]: N candidate pattern / crowding scores.
        list[bool]: N `is_text` flag. True if text was detected. False otherwise.
    """
    # get parameters
    pattern_params = score_params.get("pattern", {})
    crowding_params = score_params.get("crowding", {})
    text_overlap_params = score_params.get("text_overlap", {})
    toggle_params = score_params.get("toggle", {})

    confidences = []
    is_texts = []

    for bbox, bbox_dpi in zip(bboxes, bboxes_dpi, strict=True):
        # 1) Get scaled area
        im_gray_bbox = im_gray_dpi[int(bbox_dpi.y0) : int(bbox_dpi.y1), int(bbox_dpi.x0) : int(bbox_dpi.x1)]
        im_binary_bbox = im_binary_dpi[int(bbox_dpi.y0) : int(bbox_dpi.y1), int(bbox_dpi.x0) : int(bbox_dpi.x1)]

        # 2) Compute local stats
        confidence = 1.0
        is_text = 1.0

        if toggle_params["crowding"]:
            confidence *= _score_crowding(im_binary_bbox, **crowding_params)
        if toggle_params["pattern"]:
            confidence *= _score_pattern(im_gray_bbox, pattern_params)
        if toggle_params["text_overlap"]:
            is_text = _is_text(bbox, text_lines=text_lines, **text_overlap_params)
        # 3) Append to predictions
        confidences.append(confidence)
        is_texts.append(is_text)

    return confidences, is_texts


def _merge_sections(
    sections: list[StripLogSection],
    structures_bbox: list[pymupdf.Rect],
    merge_params: dict,
) -> list[StripLog]:
    """Merge vertically aligned and connected section candidates into contiguous striplogs.

    Starting from **seed** sections (confidence ≥ `confidence_seed`), the procedure
    grows a striplog by attaching **connected** neighbor sections whose confidence
    is ≥ `confidence_graph`. Two sections are considered connected if:
      1) they are vertically aligned with the seed within a horizontal tolerance, and
      2) there exists at least one vertical structure line (from `structures_bbox`)
         that intersects/bridges both sections (i.e., a shared connection line).

    The result is a set of clusters (striplogs), each representing a vertical stack
    of sections belonging to the same column/track.

    ASCII examples
    --------------
    We denotes as `s` the seed cells and `c` the candidates. The Adjacency is A + A.T > 0

    Result is group {2, 3}          Result is group {1, 2}
    +-------+                       +-------+
    | 1 (c) |                       | 1 (c) |
    +-------+                       +-------+
              (not connected)       |
    +-------+                       +-------+
    | 2 (s) |                       | 2 (s) |
    +-------+                       +-------+
            | (connected)                   |
    +-------+                               +-------+
    | 3 (c) |                               | 3 (c) |
    +-------+                               +-------+

    Adjacency {2, 3}                Adjacency {1, 2}
        1   2   3                        1   2   3
    1 | 0 | 0 | 0 |                 1  | 0 | 1 | 0 |
    2 | 0 | 1 | 1 |                 2  | 1 | 1 | 0 |
    3 | 0 | 1 | 0 |                 3  | 0 | 0 | 0 |

    Args:
        sections (list[StripLogSection]): Detected section candidates.
        structures_bbox (list[pymupdf.Rect]): Vertical line segments that span the page.
        merge_params (dict): Parameters relative to section merging

    Returns:
        list[StripLog]: A list of merged `StripLog` objects.
    """
    # Check if at least one section detected
    if not sections:
        return []

    # Get seed striplogs
    graph_sections = [strip for strip in sections if strip.confidence >= merge_params["confidence_graph"]]
    graph_adjacency = np.zeros((len(graph_sections), len(graph_sections)), dtype=bool)

    # Iterate over seed and check for graph connections
    for i, node in enumerate(graph_sections):
        # Check if node is potential seed
        if node.confidence < merge_params["confidence_seed"]:
            continue
        # Compute connections between seed and structure lines
        edges = [line for line in structures_bbox if line.intersects(node.bbox)]
        # Compute intersections with other (add constraint on width)
        graph_adjacency[i] = [
            any([edge.intersects(v.bbox) and node.aligns(v, merge_params["r_tol_width"]) for edge in edges])
            for v in graph_sections
        ]

    # Check if at least one cell reached confidence_seed
    if graph_adjacency.sum() == 0:
        return []

    # Ensure the matrix is symmetric (avoid undirected graph)
    graph_adjacency = ((graph_adjacency + graph_adjacency.T) > 0).astype(int)
    _, labels = connected_components(csgraph=graph_adjacency, directed=False, return_labels=True)

    # Build striplogs and prune if too small
    striplogs: list[StripLog] = []
    for c in np.unique(labels):
        cluster_indices = np.where(labels == c)[0]
        cluster_sections = np.array(graph_sections)[cluster_indices]
        if len(cluster_sections) < merge_params["min_sections"]:
            continue
        striplogs.append(StripLog.from_striplog_sections(sections=cluster_sections))

    return striplogs


def _is_ocr_artifact(text: str) -> bool:
    """Check for OCR text artifacts.

    Considers a string an artifact if it consists only of (case-insensitive) letters 'o'/'O',
    digits {0,1,8}, whitespace, and the punctuation set `|/()-.,=_`.

    Args:
        text (str): Raw text to classify.

    Returns:
        bool: True if text is composed of artifacts.
    """
    ocr_numeric_pattern = re.compile(r"^[\so018|/()\-\.,=_]+$", re.IGNORECASE)
    return bool(ocr_numeric_pattern.match(text))


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
    dpi = striplog_detection_params.get("dpi", 72)
    threshold_param = striplog_detection_params.get("threshold", {})
    table_param = striplog_detection_params.get("table", {})
    score_params = striplog_detection_params.get("score", {})
    merge_params = striplog_detection_params.get("merge", {})

    # Rendering and thresholding
    im_gray_dpi, rescale_factor = _page_to_grayscale(page=page, dpi=dpi)
    im_thresholded_dpi = _threshold_image(im_gray_dpi, **threshold_param)

    # Candidate detection in page space
    section_bboxes_dpi, structures_bbox_dpi = _detect_candidates_from_page(im_thresholded_dpi, table_param)

    # Rescale bounding boxes to match output DPI
    section_bboxes = _rescale_bboxes(section_bboxes_dpi, scale=rescale_factor)
    structures_bbox = _rescale_bboxes(structures_bbox_dpi, scale=rescale_factor)

    # Create Striplog section candidates
    confidences, is_texts = _score_candidates(
        section_bboxes, section_bboxes_dpi, im_gray_dpi, im_thresholded_dpi, score_params, text_lines
    )

    section_candidates = [
        StripLogSection(
            bbox=bbox,
            confidence=confidence,
        )
        for bbox, confidence, is_text in zip(section_bboxes, confidences, is_texts, strict=True)
        if not is_text
    ]

    # Vertical merging, then filter by minimum section count.
    return _merge_sections(section_candidates, structures_bbox, merge_params)
