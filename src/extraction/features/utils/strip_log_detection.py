"""This module contains functionalities to detect strip-log structures."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pymupdf

from extraction.features.utils.geometry.circle_detection import extract_circles
from extraction.features.utils.geometry.geometry_dataclasses import Circle, Line
from extraction.features.utils.table_detection import StructureLine, detect_structure_lines
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

logger = logging.getLogger(__name__)

config = read_params("table_detection_params.yml")


@dataclass
class StripLog:
    """Represents a detected strip log or soil profile structure."""

    bounding_rect: pymupdf.Rect
    vertical_lines: list[Line]
    horizontal_lines: list[Line]
    circles: list[Circle]
    confidence: float


def detect_strip_logs(
    page: pymupdf.Page,
    geometric_lines: list[Line],
    line_detection_params: dict,
    text_lines: list[TextLine],
) -> list[StripLog]:
    """Detect strip logs (soil profiles) on a page.

    Strip logs are characterized by:
    - Long vertical lines forming the boundaries
    - High density of short horizontal lines between verticals
    - High density of circles between verticals
    - Absence of text within the structure
    - Relatively narrow width compared to height

    Args:
        page (pymupdf.Page): The page to analyze.
        geometric_lines (list[Line]): Detected geometric lines on the page.
        line_detection_params (dict): Parameters for circle detection.
        text_lines (list[TextLine]): All text lines on the page.

    Returns:
        List of detected strip log structures
    """
    geometric_circles = extract_circles(page, line_detection_params, text_lines)
    structure_lines = detect_structure_lines(geometric_lines, filter_lines=False)
    strip_candidates = _find_strip_log_structures(structure_lines, geometric_circles, config, text_lines)

    # Filter based on strip log criteria
    filtered_strips = []
    for strip in strip_candidates:
        if _is_valid_strip_log(strip, config):
            filtered_strips.append(strip)

    return filtered_strips


def _find_strip_log_structures(
    structure_lines: list[StructureLine],
    circles: list[Circle],
    config: dict,
    text_lines: list[TextLine] = None,
) -> list[StripLog]:
    """Find strip log structures on a page.

    Args:
        structure_lines: List of structure lines
        circles: List of detected circles
        config: Strip log specific configuration parameters
        text_lines: List of text lines for content analysis

    Returns:
        List of detected strip log candidates
    """
    vertical_lines = sorted([line for line in structure_lines if line.is_vertical], key=lambda line: line.position)
    horizontal_lines = [line for line in structure_lines if not line.is_vertical]

    strip_candidates = []

    for i, first_line in enumerate(vertical_lines):
        for second_line in vertical_lines[i + 1 : i + 4]:
            # Check if lines could form a strip (reasonable distance apart)
            strip_height = max(first_line.end, second_line.end) - min(first_line.start, second_line.start)

            if strip_height == 0:
                continue

            # Create potential strip region
            strip_rect = pymupdf.Rect(
                min(first_line.position, second_line.position),
                min(first_line.start, second_line.start),
                max(first_line.position, second_line.position),
                max(first_line.end, second_line.end),
            )

            # Find horizontal lines and circles within this region
            region_horizontals = _find_horizontal_lines_in_region(horizontal_lines, strip_rect)
            region_circles = _find_circles_in_region(circles, strip_rect)

            # Create strip log candidate
            strip = _create_strip_log_from_region(
                strip_rect,
                [first_line.line, second_line.line],
                [line.line for line in region_horizontals],
                region_circles,
                config,
                text_lines,
            )

            if strip:
                strip_candidates.append(strip)

    # Remove overlapping strips, keep highest confidence
    final_strips = []
    strip_candidates = sorted(strip_candidates, key=lambda s: s.confidence, reverse=True)

    for strip in strip_candidates:
        if not any(_strip_overlaps(strip, existing) for existing in final_strips):
            final_strips.append(strip)

    return final_strips


def _find_horizontal_lines_in_region(
    horizontal_lines: list[StructureLine], region: pymupdf.Rect
) -> list[StructureLine]:
    """Find horizontal lines that intersect with a strip region.

    Args:
        horizontal_lines: List of horizontal structure lines
        region: The bounding rectangle of the strip region
    Returns:
        List of horizontal structure lines within the region
    """
    lines_in_region = []
    for line in horizontal_lines:
        # For horizontal lines, check if position is within y bounds and line overlaps x bounds
        if region.y0 <= line.position <= region.y1 and line.start <= region.x1 and line.end >= region.x0:
            lines_in_region.append(line)
    return lines_in_region


def _find_circles_in_region(circles: list[Circle], region: pymupdf.Rect) -> list[Circle]:
    """Find circles that fall within a given region."""
    circles_in_region = []
    for circle in circles:
        # Check if circle center is within the region
        if region.x0 <= circle.center.x <= region.x1 and region.y0 <= circle.center.y <= region.y1:
            circles_in_region.append(circle)
    return circles_in_region


def _create_strip_log_from_region(
    region_rect: pymupdf.Rect,
    vertical_lines: list[Line],
    horizontal_lines: list[Line],
    circles: list[Circle],
    config: dict,
    text_lines: list[TextLine] = None,
) -> StripLog | None:
    """Create a strip log structure from a detected region.

    Args:
        region_rect: Bounding rectangle of the strip region
        vertical_lines: Vertical lines defining the strip
        horizontal_lines: Horizontal lines within the strip
        circles: Circles within the strip
        config: Strip log specific configuration parameters
        text_lines: List of text lines for content analysis

    Returns:
        StripLog object or None if criteria not met
    """
    if not vertical_lines:
        return None

    # Calculate confidence
    confidence = _calculate_strip_confidence(region_rect, horizontal_lines, circles, config, text_lines)

    return StripLog(
        bounding_rect=region_rect,
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
        circles=circles,
        confidence=confidence,
    )


def _calculate_strip_confidence(
    rect: pymupdf.Rect,
    horizontal_lines: list[Line],
    circles: list[Circle],
    config: dict,
    text_lines: list[TextLine] = None,
) -> float:
    """Calculate confidence score for a strip log structure.

    Args:
        rect: Bounding rectangle of the strip log
        horizontal_lines: Horizontal lines within the strip
        circles: Circles within the strip
        config: Strip log specific configuration parameters
        text_lines: List of text lines for content analysis

    Returns:
        Confidence score between 0 and 1
    """
    strip_config = config.get("strip_logs", {})
    area = rect.width * rect.height

    # Aspect ratio score (height/width)
    if rect.width < strip_config.get("min_width"):
        return 0.0

    else:
        aspect_scoring = strip_config.get("aspect")
        aspect_ratio = rect.height / rect.width
        aspect_score = min(aspect_scoring.get("aspect_weight"), aspect_ratio / aspect_scoring.get("min_aspect_ratio"))

        # Horizontal line density score
        line_scoring = strip_config.get("line_density")
        h_line_density = len(horizontal_lines) / (area / 10000)
        line_density = min(
            line_scoring.get("line_density_weight"), h_line_density / line_scoring.get("min_horizontal_density")
        )

        # Circle density score
        circle_scoring = strip_config.get("circle_density")
        circle_density = len(circles) / (area / 10000)
        circle_score = min(
            circle_scoring.get("circle_weight"), circle_density / circle_scoring.get("min_circle_density")
        )

        # Penalty for text within the region (strip logs should have no text)
        text_penalty = 0.0
        if text_lines:
            text_within = [line for line in text_lines if rect.intersects(line.rect)]
            meaningful_texts = []

            for line in text_within:
                if line.text.strip() and not _is_numeric_pattern(line.text.strip()):
                    meaningful_texts.append(line)

            if meaningful_texts:
                text_penalty = len(meaningful_texts) * strip_config.get("text_penalty")

        confidence = aspect_score + line_density + circle_score - text_penalty

    return min(1.0, confidence)


def _is_numeric_pattern(text_content: str) -> bool:
    """Check if text matches common numeric patterns in strip logs."""
    # Pattern for combinations of digits and dots (0, 0.0, 00, 0.0.0, 08, .00, etc.)
    numeric_pattern = re.compile(r"^[\s08.]+$")

    return bool(numeric_pattern.match(text_content))


def _is_valid_strip_log(strip: StripLog, config: dict) -> bool:
    """Check if a strip log candidate meets the minimum criteria."""
    strip_config = config.get("strip_logs", {})
    min_confidence = strip_config.get("min_confidence")
    min_horizontal_lines = strip_config.get("min_horizontal_lines")

    return (
        strip.confidence >= min_confidence
        and len(strip.horizontal_lines) >= min_horizontal_lines
        and len(strip.vertical_lines) >= 2
    )


def _strip_overlaps(strip1: StripLog, strip2: StripLog) -> bool:
    """Check if two strip logs overlap significantly."""
    return strip1.bounding_rect.intersects(strip2.bounding_rect)
