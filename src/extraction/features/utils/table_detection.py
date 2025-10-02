"""This module contains functionalities to detect table like structures."""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass

import pymupdf

from extraction.features.stratigraphy.layer.page_bounding_boxes import MaterialDescriptionRectWithSidebar
from extraction.features.utils.geometry.geometry_dataclasses import Circle, Line
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

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


def detect_structure_lines(geometric_lines: list[Line]) -> list[StructureLine]:
    """Detect significant horizonal and vertical lines in a document.

    Args:
        geometric_lines (list[Line]): Geometric lines (e.g., from layout analysis).

    Returns:
        List of detected structure lines
    """
    # Filter and classify lines
    filtered_lines = _filter_significant_lines(geometric_lines, config)
    return _separate_by_orientation(filtered_lines, config)


def detect_table_structures(
    page_index: int,
    document: pymupdf.Document,
    structure_lines: list[StructureLine],
    text_lines: list[TextLine],
) -> list[TableStructure]:
    """Detect multiple non-overlapping table structures on a page.

    Args:
        page_index (int): The page index (0-indexed).
        document (pymupdf.Document): the document.
        structure_lines (list[StructureLine]): Vertical and horizonal structure lines.
        text_lines (list[TextLine]): All text lines on the page.

    Returns:
        List of detected table structures
    """
    # Get page dimensions from the document
    page = document[page_index]
    page_width = page.rect.width
    page_height = page.rect.height

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


def _contained_in_table_index(
    pair: MaterialDescriptionRectWithSidebar, table_structures: list[TableStructure], proximity_buffer: float = 50
) -> int:
    """Returns the index of the first table structure that contains this pair, or -1 if none is found.

    Args:
        pair: MaterialDescriptionRectWithSidebar object
        table_structures: List of table structures
        proximity_buffer: Distance threshold for proximity check

    Returns:
        The index of the first table structure that contains this pair, or -1 if none is found
    """
    material_rect = pair.material_description_rect
    sidebar_rect = pair.sidebar.rect if pair.sidebar else None

    for index, table in enumerate(table_structures):
        # Check if rectangle is within proximity buffer of table
        expanded_table_rect = pymupdf.Rect(
            table.bounding_rect.x0 - proximity_buffer,
            table.bounding_rect.y0 - proximity_buffer,
            table.bounding_rect.x1 + proximity_buffer,
            table.bounding_rect.y1 + proximity_buffer,
        )

        material_rect_inside = expanded_table_rect.contains(material_rect)
        sidebar_rect_inside = expanded_table_rect.contains(sidebar_rect) if sidebar_rect else True

        if material_rect_inside and sidebar_rect_inside:
            return index

    return -1


@dataclass
class StripLog:
    """Represents a detected strip log or soil profile structure."""

    bounding_rect: pymupdf.Rect
    vertical_lines: list[Line]
    horizontal_lines: list[Line]
    circles: list[Circle]
    confidence: float


def detect_strip_logs(
    structure_lines: list[StructureLine],
    circles: list[Circle],
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
        page_index (int): The page index (0-indexed).
        document (pymupdf.Document): The document.
        structure_lines (list[StructureLine]): Vertical and horizontal structure lines.
        circles (list[Circle]): Detected circles on the page.
        text_lines (list[TextLine]): All text lines on the page.

    Returns:
        List of detected strip log structures
    """
    strip_candidates = _find_strip_log_structures(structure_lines, circles, config, text_lines)

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
        for offset in range(1, 3):
            j = i + offset
            if j >= len(vertical_lines):
                break
            second_line = vertical_lines[j]

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
        confidence = 0
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
    # Remove all whitespace for pattern matching
    clean_text = text_content.replace(" ", "")

    # Pattern for combinations of digits and dots (0, 0.0, 00, 0.0.0, 08, .00, etc.)
    numeric_pattern = re.compile(r"^[08.]+$")

    return bool(numeric_pattern.match(clean_text))


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
