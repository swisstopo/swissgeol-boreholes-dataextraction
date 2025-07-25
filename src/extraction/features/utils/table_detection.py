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
    """Detect large table structures on a page.

    Args:
        geometric_lines: List of detected geometric lines
        page_width: Page width
        page_height: Page height
        text_lines: List of text lines for text content analysis

    Returns:
        List containing at most one large table structure
    """
    config = read_params("table_detection_params.yml")

    # Filter and classify lines
    filtered_lines = _filter_significant_lines(geometric_lines, config, page_width, page_height)
    structure_lines = _separate_by_orientation(filtered_lines, config)

    # Find the dominant table structure
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


def _filter_significant_lines(
    lines: list[Line], config: dict, page_width: float = None, page_height: float = None
) -> list[Line]:
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
        # TODO docstring and unit tests
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
    """Find the all table structures on the page."""
    line_partitions = []

    while lines:
        current_partition = []
        unprocessed = [lines.pop()]
        while unprocessed:
            current_line = unprocessed.pop()
            remaining_lines = []
            for line in lines:
                if current_line.connects_with(line):
                    unprocessed.append(line)
                else:
                    remaining_lines.append(line)
            current_partition.append(current_line)
            lines = remaining_lines
        line_partitions.append(current_partition)

    return [_create_table_structure(lines, config, page_width, page_height, text_lines) for lines in line_partitions]


def _create_table_structure(
    lines: list[StructureLine],
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: list = None,
) -> TableStructure:
    """Create a table structure from a collection of connected lines."""
    all_lines: list[Line] = [line.line for line in lines]
    horizontal_lines: list[Line] = [line.line for line in lines if not line.is_vertical]
    vertical_lines: list[Line] = [line.line for line in lines if line.is_vertical]

    # Calculate bounding box of all lines
    min_x = min(min(line.start.x, line.end.x) for line in all_lines)
    max_x = max(max(line.start.x, line.end.x) for line in all_lines)
    min_y = min(min(line.start.y, line.end.y) for line in all_lines)
    max_y = max(max(line.start.y, line.end.y) for line in all_lines)

    # Create initial bounding rectangle
    bounding_rect = pymupdf.Rect(min_x, min_y, max_x, max_y)

    # Calculate metrics
    area = bounding_rect.width * bounding_rect.height
    # normalize the area
    line_density = len(horizontal_lines + vertical_lines) / (area / 10000) if area > 0 else 0

    # Calculate confidence based on structure quality
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
    """Calculate confidence score for the table structure."""
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


def _pair_conflicts_with_tables(
    pair: "MaterialDescriptionRectWithSidebar", table_structures: list[TableStructure], proximity_buffer: float = 50
) -> bool:
    """Check if a material description pair is relevant to any table structure.

    Args:
        pair: MaterialDescriptionRectWithSidebar object
        table_structures: List of table structures
        proximity_buffer: Distance threshold for proximity check

    Returns:
        True if pair is inside or near any table structure
    """
    material_rect = pair.material_description_rect
    sidebar_rect = pair.sidebar.rect() if pair.sidebar else None

    for table in table_structures:
        # Check if rectangle is within proximity buffer of table
        expanded_table_rect = pymupdf.Rect(
            table.bounding_rect.x0 - proximity_buffer,
            table.bounding_rect.y0 - proximity_buffer,
            table.bounding_rect.x1 + proximity_buffer,
            table.bounding_rect.y1 + proximity_buffer,
        )
        shrunk_table_rect = pymupdf.Rect(
            table.bounding_rect.x0 + proximity_buffer,
            table.bounding_rect.y0 + proximity_buffer,
            table.bounding_rect.x1 - proximity_buffer,
            table.bounding_rect.y1 - proximity_buffer,
        )

        material_rect_inside = expanded_table_rect.contains(material_rect)
        material_rect_outside = not shrunk_table_rect.intersects(material_rect)
        if not (material_rect_inside or material_rect_outside):
            return True

        # Note: we currently allow the material rect to be inside the table and the sidebar rect to be outside,
        # or vice versa. Otherwise, we get bad results for e.g. 267124180-bp.pdf.
        if sidebar_rect:
            sidebar_rect_inside = expanded_table_rect.contains(sidebar_rect)
            sidebar_rect_outside = not shrunk_table_rect.intersects(sidebar_rect)
            if not (sidebar_rect_inside or sidebar_rect_outside):
                return True

    # no conflict
    return False


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
