"""This module contains functionalities to detect table like structures."""

import dataclasses
import logging
from dataclasses import dataclass

import pymupdf

from extraction.features.stratigraphy.layer.page_bounding_boxes import MaterialDescriptionRectWithSidebar
from extraction.features.utils.geometry.geometry_dataclasses import Line
from utils.file_utils import read_params

logger = logging.getLogger(__name__)


@dataclass
class TableStructure:
    """Represents a detected table structure."""

    bounding_rect: pymupdf.Rect
    horizontal_lines: list[Line]
    vertical_lines: list[Line]
    confidence: float
    line_density: float


def detect_table_structures(
    geometric_lines: list[Line], page_width: float = None, page_height: float = None, text_lines: list = None
) -> list[TableStructure]:
    """Detect multiple non-overlapping table structures on a page.

    Args:
        geometric_lines: List of detected geometric lines
        page_width: Page width
        page_height: Page height
        text_lines: List of text lines for text content analysis

    Returns:
        List of detected table structures
    """
    config = read_params("table_detection_params.yml")

    # Filter and classify lines
    filtered_lines = _filter_significant_lines(geometric_lines, config)
    structure_lines = _separate_by_orientation(filtered_lines, config)

    table_candidates = _find_table_structures(structure_lines, config, page_width, page_height, text_lines)
    table_candidates = [
        table
        for table in table_candidates
        if len(table.horizontal_lines) >= 3
        if len(table.vertical_lines) >= 1
        if table.confidence >= config.get("min_confidence")
    ]
    for table in table_candidates:
        logger.info(f"Detected table structure (confidence: {table.confidence:.3f})")
    return table_candidates


def _filter_significant_lines(lines: list[Line], config: dict) -> list[Line]:
    """Filter to keep only significantly long lines that could form table structures."""
    min_length = config.get("min_line_length")

    return [line for line in lines if line.length > min_length]


def _separate_by_orientation(lines: list[Line], config: dict) -> list["StructureLine"]:
    """Separate lines into horizontal and vertical based on angle and tolerance."""
    angle_tolerance = config.get("angle_tolerance")
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

    def connects_with(self, other: "StructureLine", tolerance: float = 10.0) -> bool:
        """Check if this line connects with another line within a given tolerance.

        Args:
            other: Another StructureLine to check connection with
            tolerance: Distance threshold for connection

        Returns:
            bool: True if lines connect, False otherwise
        """
        if self.is_vertical == other.is_vertical:
            position_ok = abs(self.position - other.position) < tolerance
            min_self = min(self.start, self.end)
            max_self = max(self.start, self.end)
            min_other = min(other.start, other.end)
            max_other = max(other.start, other.end)
            start_end_ok = not (max_self + tolerance < min_other) and not (max_other + tolerance < min_self)
            return position_ok and start_end_ok
        else:
            return (self.start - tolerance < other.position < self.end + tolerance) and (
                other.start - tolerance < self.position < other.end + tolerance
            )


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

        # Check for bounding box intersection
        if table and table.confidence >= config.get("min_confidence") and not _table_overlaps(table, detected_tables):
            detected_tables.append(table)

    return detected_tables


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
    connection_threshold = config.get("connection_threshold")

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

    # Size score - larger tables are more likely to be significant
    if page_width and page_height:
        page_area = page_width * page_height
        area_ratio = area / page_area
        area_scoring = config.get("area_scoring", {})
        size_score = min(area_scoring.get("area_weights"), area_ratio / area_scoring.get("min_table_area_ratio"))
    else:
        # Fallback to 0 if no page dimensions are available
        size_score = 0

    # Line structure score - more lines indicate better structure
    line_scoring = config.get("line_scoring", {})
    total_lines = len(h_lines) + len(v_lines)
    line_score = min(line_scoring.get("line_weights"), total_lines / line_scoring.get("max_n_lines_bonus"))

    # Text bonus score - bonus for text content within the table structure
    text_bonus = 0.0
    if text_lines:
        text_within = [line for line in text_lines if rect.intersects(line.rect)]
        if text_within:
            text_scoring = config.get("text_scoring", {})
            text_bonus = min(
                text_scoring.get("text_weights"), len(text_within) * text_scoring.get("text_presence_weight")
            )

    # Weighted combination
    total_confidence = size_score + line_score + text_bonus

    return min(1.0, total_confidence)


def _contained_in_table_index(
    pair: "MaterialDescriptionRectWithSidebar", table_structures: list[TableStructure], proximity_buffer: float = 50
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
    sidebar_rect = pair.sidebar.rect() if pair.sidebar else None

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
