"""This module contains functionalities to detect table like structures."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params

logger = logging.getLogger(__name__)

config = read_params("table_detection_params.yml")


@dataclass
class TableStructure:
    """Represents a detected table structure."""

    bounding_rect: pymupdf.Rect
    horizontal_lines: list[Line]
    vertical_lines: list[Line]
    confidence: float
    line_density: float


def detect_structure_lines(geometric_lines: list[Line], filter_lines=True) -> list[StructureLine]:
    """Detect significant horizonal and vertical lines in a document.

    Args:
        geometric_lines (list[Line]): Geometric lines (e.g., from layout analysis).
        filter_lines (bool, optional): Whether to filter lines before classification. Defaults to True.

    Returns:
        List of detected structure lines
    """
    # Filter and classify lines
    final_lines = _filter_significant_lines(geometric_lines, config) if filter_lines else geometric_lines
    return _separate_by_orientation(final_lines, config)


def detect_table_structures(
    page_index: int,
    document: pymupdf.Document,
    geometric_lines: list[Line],
    text_lines: list[TextLine],
) -> list[TableStructure]:
    """Detect multiple non-overlapping table structures on a page.

    Args:
        page_index (int): The page index (0-indexed).
        document (pymupdf.Document): the document.
        geometric_lines (list[Line]): The geometric lines on the page.
        text_lines (list[TextLine]): All text lines on the page.

    Returns:
        List of detected table structures
    """
    # Get page dimensions from the document
    page = document[page_index]
    page_width = page.rect.width
    page_height = page.rect.height

    structure_lines = detect_structure_lines(geometric_lines)
    table_candidates = _find_table_structures(structure_lines, config, page_width, page_height, text_lines)
    table_candidates = [
        table
        for table in table_candidates
        if len(table.horizontal_lines) >= 2
        if len(table.vertical_lines) >= 1
        if table.confidence >= config["tables"]["min_confidence"]
    ]
    for table in table_candidates:
        logger.debug(f"Detected table structure (confidence: {table.confidence:.3f})")
    return table_candidates


def _filter_significant_lines(lines: list[Line], config: dict) -> list[Line]:
    """Filter to keep only significantly long lines that could form table structures."""
    min_length = config["tables"]["min_line_length"]

    return [line for line in lines if line.length > min_length]


def _separate_by_orientation(lines: list[Line], config: dict) -> list[StructureLine]:
    """Separate lines into horizontal and vertical based on angle and tolerance."""
    angle_tolerance = config["tables"]["angle_tolerance"]
    structure_lines = []

    for line in lines:
        angle = abs(line.angle)

        # Horizontal lines (close to 0° or 180°)
        if angle <= angle_tolerance or angle >= (180 - angle_tolerance):
            structure_lines.append(
                StructureLine(
                    start=min(line.start.x, line.end.x),
                    end=max(line.start.x, line.end.x),
                    position=(line.start.y + line.end.y) / 2,
                    is_vertical=False,
                    line=line,
                )
            )
        # Vertical lines (close to 90°)
        elif angle - 90 <= angle_tolerance:
            structure_lines.append(
                StructureLine(
                    start=min(line.start.y, line.end.y),
                    end=max(line.start.y, line.end.y),
                    position=(line.start.x + line.end.x) / 2,
                    is_vertical=True,
                    line=line,
                )
            )

    return structure_lines


@dataclasses.dataclass
class StructureLine:
    """Helper class for representing horizontal and vertical lines in a table structure."""

    start: float
    end: float
    position: float
    is_vertical: bool
    line: Line


def _find_table_structures(
    lines: list[StructureLine],
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: list = None,
) -> list[TableStructure]:
    """Find multiple non-intersecting table structures using region-based detection.

    Args:
        lines: List of structure lines
        config: Configuration parameters
        page_width: Page width
        page_height: Page height
        text_lines: List of text lines for content analysis

    Returns:
        List of table structures
    """
    # Find line groups using region grouping
    table_regions = _find_table_regions(lines, config)

    detected_tables = []
    for region_h_lines, region_v_lines in table_regions:
        # define a minimum structure for a table
        if len(region_h_lines) < 2 or len(region_v_lines) < 2:
            continue

        # Create table structure for this region
        table = _create_table_from_region(region_h_lines, region_v_lines, config, page_width, page_height, text_lines)
        if table.confidence >= config["tables"]["min_confidence"]:
            detected_tables.append(table)

    detected_tables = sorted(detected_tables, key=lambda t: t.confidence, reverse=True)

    final_tables = []
    for table in detected_tables:
        # Check if this table overlaps with any already accepted table
        if not _table_overlaps(table, final_tables):
            final_tables.append(table)

    return final_tables


def _find_table_regions(lines: list[StructureLine], config: dict) -> list[tuple[list[Line], list[Line]]]:
    """Find regions of connected lines that could form table structures.

    Args:
        lines: List of structure lines
        config: Configuration parameters

    Returns:
        List of tuples containing horizontal and vertical lines for each detected region
    """
    line_groups = []

    for line in lines:
        # Find which existing groups this line should join
        matching_groups = []

        for group_idx, group in enumerate(line_groups):
            if _line_connects_to_group(line, group, config):
                matching_groups.append(group_idx)

        if not matching_groups:
            # Create new group
            line_groups.append([line])
        elif len(matching_groups) == 1:
            # Add to existing group
            line_groups[matching_groups[0]].append(line)
        else:
            # Merge multiple groups and add line
            merged_group = [line]
            for group_idx in sorted(matching_groups, reverse=True):
                merged_group.extend(line_groups.pop(group_idx))
            line_groups.append(merged_group)

    # Convert groups to regions with structure requirements
    regions = []
    for group in line_groups:
        group_h_lines = [line.line for line in group if not line.is_vertical]
        group_v_lines = [line.line for line in group if line.is_vertical]

        if len(group_h_lines) >= 1 and len(group_v_lines) >= 1:
            regions.append((group_h_lines, group_v_lines))

    return regions


def _line_connects_to_group(line: StructureLine, group: list[StructureLine], config: dict) -> bool:
    """Connection check combining line intersection, endpoint proximity and T-junction check.

    Args:
        line: The line to check
        group: The group of lines to check against
        config: Configuration parameters
    Returns:
        True if the line connects to the group, False otherwise
    """
    connection_threshold = config["tables"]["connection_threshold"]

    for group_line in group:
        # 1. Check line intersection
        if line.line.intersects_with(group_line.line):
            return True

        # 2. Check endpoint proximity with Euclidean distance
        for p1 in [line.line.start, line.line.end]:
            for p2 in [group_line.line.start, group_line.line.end]:
                if p1.distance_to(p2) <= connection_threshold:
                    return True

        # 3. Check T-junction (point near line)
        for point in [line.line.start, line.line.end]:
            if group_line.line.point_near_segment(point, connection_threshold):
                return True

        for point in [group_line.line.start, group_line.line.end]:
            if line.line.point_near_segment(point, connection_threshold):
                return True

    return False


def _table_overlaps(table: TableStructure, existing_tables: list[TableStructure]) -> bool:
    """Overlap check using bounding box intersection."""
    return any(table.bounding_rect.intersects(existing_table.bounding_rect) for existing_table in existing_tables)


def _create_table_from_region(
    horizontal_lines: list[Line],
    vertical_lines: list[Line],
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: list = None,
) -> TableStructure | None:
    """Create a table structure from a region of connected lines.

    Args:
        horizontal_lines: Horizontal lines in this region
        vertical_lines: Vertical lines in this region
        config: Configuration parameters
        page_width: Page width
        page_height: Page height
        text_lines: Text lines for content analysis

    Returns:
        TableStructure object or None
    """
    all_lines = horizontal_lines + vertical_lines
    if not all_lines:
        return None

    # Calculate bounding box of this region
    min_x = min(min(line.start.x, line.end.x) for line in all_lines)
    max_x = max(max(line.start.x, line.end.x) for line in all_lines)
    min_y = min(min(line.start.y, line.end.y) for line in all_lines)
    max_y = max(max(line.start.y, line.end.y) for line in all_lines)

    # Create bounding rectangle for this specific region
    bounding_rect = pymupdf.Rect(min_x, min_y, max_x, max_y)

    # Calculate metrics
    area = bounding_rect.width * bounding_rect.height
    # Normalize the area
    line_density = len(horizontal_lines + vertical_lines) / (area / 10000) if area > 0 else 0

    confidence = _calculate_structure_confidence(
        bounding_rect, horizontal_lines, vertical_lines, config, page_width, page_height, text_lines
    )

    return TableStructure(
        bounding_rect=bounding_rect,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        confidence=confidence,
        line_density=line_density,
    )


def _calculate_structure_confidence(
    rect: pymupdf.Rect,
    h_lines: list[Line],
    v_lines: list[Line],
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: list = None,
) -> float:
    """Calculate confidence score for the table structure.

    Args:
        rect: Bounding rectangle of the table structure
        h_lines: List of horizontal lines in the structure
        v_lines: List of vertical lines in the structure
        config: Configuration parameters
        page_width: Page width
        page_height: Page height
        text_lines: List of text lines for content analysis

    Returns:
        Confidence score between 0 and 1
    """
    area = rect.width * rect.height
    table_config = config.get("tables", {})

    # Size score - larger tables are more likely to be significant
    if page_width and page_height:
        page_area = page_width * page_height
        area_ratio = area / page_area
        area_scoring = table_config.get("area_scoring", {})
        size_score = min(area_scoring.get("area_weights"), area_ratio / area_scoring.get("min_table_area_ratio"))
    else:
        # Fallback to 0 if no page dimensions are available
        size_score = 0

    # Line structure score - more lines indicate better structure
    line_scoring = table_config.get("line_scoring", {})
    total_lines = len(h_lines) + len(v_lines)
    line_score = min(line_scoring.get("line_weights"), total_lines / line_scoring.get("max_n_lines_bonus"))

    # Text bonus score - bonus for text content within the table structure
    text_bonus = 0.0
    if text_lines:
        text_within = [line for line in text_lines if rect.intersects(line.rect)]
        if text_within:
            text_scoring = table_config.get("text_scoring", {})
            text_bonus = min(
                text_scoring.get("text_weights"), len(text_within) * text_scoring.get("text_presence_weight")
            )

    # Weighted combination
    total_confidence = size_score + line_score + text_bonus

    return min(1.0, total_confidence)
