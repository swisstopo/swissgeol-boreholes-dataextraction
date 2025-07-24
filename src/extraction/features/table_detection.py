"""This module contains functionalities to detect table like structures"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Point, Line

from utils.file_utils import read_params

from extraction.features.stratigraphy.layer.page_bounding_boxes import MaterialDescriptionRectWithSidebar

logger = logging.getLogger(__name__)


@dataclass
class TableStructure:
    """Represents a detected table structure."""
    bounding_rect: pymupdf.Rect
    horizontal_lines: List[Line]
    vertical_lines: List[Line]
    confidence: float
    line_density: float

    def contains_rect(self, rect: pymupdf.Rect) -> bool:
        """Check if a rectangle is within this table structure."""
        return (self.bounding_rect.x0 <= rect.x0 and rect.x1 <= self.bounding_rect.x1 and
                self.bounding_rect.y0 <= rect.y0 and rect.y1 <= self.bounding_rect.y1)

    def to_json(self) -> dict:
        """Convert the TableStructure object to a JSON serializable format."""
        return {
            "bounding_rect": [
                self.bounding_rect.x0,
                self.bounding_rect.y0,
                self.bounding_rect.x1,
                self.bounding_rect.y1
            ],
            "horizontal_lines": [
                [[line.start.x, line.start.y], [line.end.x, line.end.y]]
                for line in self.horizontal_lines
            ],
            "vertical_lines": [
                [[line.start.x, line.start.y], [line.end.x, line.end.y]]
                for line in self.vertical_lines
            ],
            "confidence": self.confidence,
            "line_density": self.line_density
        }

    @classmethod
    def from_json(cls, data: dict) -> "TableStructure":
        """Create a TableStructure object from a JSON dictionary."""

        # Reconstruct horizontal lines
        horizontal_lines = [
            Line(start=Point(coords[0][0], coords[0][1]), end=Point(coords[1][0], coords[1][1]))
            for coords in data["horizontal_lines"]
        ]

        # Reconstruct vertical lines
        vertical_lines = [
            Line(start=Point(coords[0][0], coords[0][1]), end=Point(coords[1][0], coords[1][1]))
            for coords in data["vertical_lines"]
        ]

        return cls(
            bounding_rect=pymupdf.Rect(*data["bounding_rect"]),
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            confidence=data["confidence"],
            line_density=data["line_density"]
        )


def detect_table_structures(
    geometric_lines: List[Line],
    page_width: float = None,
    page_height: float = None,
    text_lines: List = None
) -> List[TableStructure]:
    """Detect large table structures on a page.

    Args:
        geometric_lines: List of detected geometric lines
        page_width: Page width 
        page_height: Page height
        text_lines: List of text lines for text content analysis

    Returns:
        List containing at most one large table structure
    """
    config = read_params('table_detection_params.yml')

    # Filter and classify lines
    filtered_lines = _filter_significant_lines(geometric_lines, config, page_width, page_height)
    horizontal_lines, vertical_lines = _separate_by_orientation(filtered_lines, config)

    # Need substantial line structure for a table
    if len(horizontal_lines) < 4 or len(vertical_lines) < 2:
        return []

    # Find the dominant table structure
    table_candidate = _find_dominant_table_structure(
        horizontal_lines, vertical_lines, config, page_width, page_height, text_lines
    )

    if table_candidate and table_candidate.confidence >= config.get('min_confidence'):
        logger.info(f"Detected table structure (confidence: {table_candidate.confidence:.3f})")
        return [table_candidate]

    logger.info("No significant table structure found")
    return []


def _filter_significant_lines(
    lines: List[Line], 
    config: dict, 
    page_width: float = None, 
    page_height: float = None
) -> List[Line]:
    """Filter to keep only significantly long lines that could form table structures."""
    min_length = config.get('min_line_length')
    margin = config.get('page_boundary_margin')

    filtered_lines = []

    for line in lines:
        # Skip very short lines
        if line.length < min_length:
            continue

        # Skip boundary lines if page dimensions are known
        if page_width and page_height:
            if (line.start.x <= margin or line.end.x <= margin or 
                line.start.x >= page_width - margin or line.end.x >= page_width - margin or
                line.start.y <= margin or line.end.y <= margin or 
                line.start.y >= page_height - margin or line.end.y >= page_height - margin):
                continue

        filtered_lines.append(line)

    return filtered_lines


def _separate_by_orientation(lines: List[Line], config: dict) -> Tuple[List[Line], List[Line]]:
    """Separate lines into horizontal and vertical based on angle and tolerance."""
    angle_tolerance = config.get('angle_tolerance')
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        angle = abs(line.angle) 

        # Horizontal lines (close to 0° or 180°)
        if angle <= angle_tolerance or angle >= (180 - angle_tolerance):
            horizontal_lines.append(line)
        # Vertical lines (close to 90°)
        elif angle - 90 <= angle_tolerance:
            vertical_lines.append(line)

    return horizontal_lines, vertical_lines


def _find_dominant_table_structure(
    horizontal_lines: List[Line],
    vertical_lines: List[Line], 
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: List = None
) -> Optional[TableStructure]:
    """Find the single dominant table structure on the page."""

    # Find the overall bounding box of all significant lines
    all_lines = horizontal_lines + vertical_lines
    if not all_lines:
        return None

    # Calculate bounding box of all lines
    min_x = min(min(line.start.x, line.end.x) for line in all_lines)
    max_x = max(max(line.start.x, line.end.x) for line in all_lines)
    min_y = min(min(line.start.y, line.end.y) for line in all_lines)
    max_y = max(max(line.start.y, line.end.y) for line in all_lines)

    # Create initial bounding rectangle
    bounding_rect = pymupdf.Rect(min_x, min_y, max_x, max_y)

    # Refine the rectangle to focus on the densest region
    refined_rect = _refine_table_bounds(bounding_rect, horizontal_lines, vertical_lines)

    # Filter lines that contribute to this table
    table_horizontal_lines = _get_lines_in_rect(horizontal_lines, refined_rect, is_horizontal=True)
    table_vertical_lines = _get_lines_in_rect(vertical_lines, refined_rect, is_horizontal=False)

    # Calculate metrics
    area = refined_rect.width * refined_rect.height
    line_density = len(table_horizontal_lines + table_vertical_lines) / (area / 10000) # normalize the area

    # Calculate confidence based on structure quality
    confidence = _calculate_structure_confidence(
        refined_rect, table_horizontal_lines, table_vertical_lines, config, page_width, page_height, text_lines
    )

    return TableStructure(
        bounding_rect=refined_rect,
        horizontal_lines=table_horizontal_lines,
        vertical_lines=table_vertical_lines,
        confidence=confidence,
        line_density=line_density
    )


def _refine_table_bounds(
    initial_rect: pymupdf.Rect,
    horizontal_lines: List[Line],
    vertical_lines: List[Line]
) -> pymupdf.Rect:
    """Refine the table bounds to focus on the main structure."""

    # Find the main vertical boundaries (longest or most central vertical lines)
    if len(vertical_lines) >= 2:
        # Sort by length and position to find structural lines
        v_sorted = sorted(vertical_lines, key=lambda l: l.length, reverse=True)
        main_verticals = v_sorted[:min(10, len(v_sorted))]  # Top 10 longest

        # Get x-coordinates of main vertical lines
        x_positions = []
        for line in main_verticals:
            x_pos = (line.start.x + line.end.x) / 2
            x_positions.append(x_pos)

        if x_positions:
            refined_min_x = min(x_positions) - 10  # Small buffer
            refined_max_x = max(x_positions) + 10. # Small buffer
        else:
            refined_min_x = initial_rect.x0
            refined_max_x = initial_rect.x1
    else:
        refined_min_x = initial_rect.x0
        refined_max_x = initial_rect.x1

    # Find the main horizontal boundaries
    if len(horizontal_lines) >= 2:
        # Sort by position to find top and bottom structural lines
        h_sorted = sorted(horizontal_lines, key=lambda l: (l.start.y + l.end.y) / 2)
        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]

        refined_min_y = min(top_line.start.y, top_line.end.y) - 5
        refined_max_y = max(bottom_line.start.y, bottom_line.end.y) + 5
    else:
        refined_min_y = initial_rect.y0
        refined_max_y = initial_rect.y1

    return pymupdf.Rect(refined_min_x, refined_min_y, refined_max_x, refined_max_y)


def _get_lines_in_rect(
    lines: List[Line],
    rect: pymupdf.Rect,
    is_horizontal: bool,
    tolerance: float = 10.0
) -> List[Line]:
    """Get lines that intersect with or are contained in the rectangle."""
    region_lines = []

    for line in lines:
        if is_horizontal:
            # For horizontal lines, check if they span across the rect width
            line_y = (line.start.y + line.end.y) / 2
            if (rect.y0 - tolerance <= line_y <= rect.y1 + tolerance and
                max(line.start.x, line.end.x) >= rect.x0 and
                min(line.start.x, line.end.x) <= rect.x1):
                region_lines.append(line)
        else:
            # For vertical lines, check if they span across the rect height
            line_x = (line.start.x + line.end.x) / 2
            if (rect.x0 - tolerance <= line_x <= rect.x1 + tolerance and
                max(line.start.y, line.end.y) >= rect.y0 and
                min(line.start.y, line.end.y) <= rect.y1):
                region_lines.append(line)

    return region_lines


def _calculate_structure_confidence(
    rect: pymupdf.Rect,
    h_lines: List[Line],
    v_lines: List[Line],
    config: dict,
    page_width: float = None,
    page_height: float = None,
    text_lines: List = None
) -> float:
    """Calculate confidence score for the table structure."""

    area = rect.width * rect.height

    # Size score - larger tables are more likely to be significant
    if page_width and page_height:
        page_area = page_width * page_height
        area_ratio = area / page_area
        area_scoring = config.get('area_scoring', {})
        size_score = min(
            area_scoring.get('area_weights'),
            area_ratio / area_scoring.get('min_table_area_ratio')
        )
    else:
        # Fallback to 0 if no page dimensions are available
        size_score = 0

    # Line structure score - more lines indicate better structure
    line_scoring = config.get('line_scoring', {})
    total_lines = len(h_lines) + len(v_lines)
    line_score = min(
        line_scoring.get('line_weights'),
        total_lines / line_scoring.get('max_n_lines_bonus')
    )

    # Text bonus score - bonus for text content within the table structure
    text_bonus = 0.0
    if text_lines:
        text_within = [line for line in text_lines if rect.intersects(line.rect)]
        if text_within:
            text_scoring = config.get('text_scoring', {})
            text_bonus = min(
                text_scoring.get('text_weights'),
                len(text_within) * text_scoring.get('text_presence_weight')
            )

    # Weighted combination
    total_confidence = (size_score + line_score + text_bonus)

    return min(1.0, total_confidence)



def filter_extraction_pairs_by_tables(
    material_description_pairs: List["MaterialDescriptionRectWithSidebar"],
    table_structures: List[TableStructure],
    proximity_buffer: float = 50.0
) -> List["MaterialDescriptionRectWithSidebar"]:
    """Filter material description pairs by table structures.

    Keeps pairs that are either inside table structures or within proximity of them.
    Falls back to all pairs if no table structures are detected.

    Args:
        material_description_pairs: List of MaterialDescriptionRectWithSidebar objects
        table_structures: List of detected table structures
        proximity_buffer: Distance in pixels for "close to table" consideration

    Returns:
        Filtered list of material description pairs
    """

    if not material_description_pairs:
        return material_description_pairs

    filtered_pairs = []

    for pair in material_description_pairs:
        if _is_pair_relevant_to_tables(pair, table_structures, proximity_buffer):
            filtered_pairs.append(pair)

    return filtered_pairs


def _is_pair_relevant_to_tables(
    pair: "MaterialDescriptionRectWithSidebar",
    table_structures: List[TableStructure],
    proximity_buffer: float
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
        # Check if material description is inside or near table
        if _is_rect_relevant_to_table(material_rect, table, proximity_buffer):
            return True

        # Check if sidebar is inside or near table
        if sidebar_rect and _is_rect_relevant_to_table(sidebar_rect, table, proximity_buffer):
            return True

    return False


def _is_rect_relevant_to_table(
    rect: pymupdf.Rect,
    table: TableStructure,
    proximity_buffer: float
) -> bool:
    """Check if a rectangle is inside or near a table structure.

    Args:
        rect: Rectangle to check
        table: Table structure
        proximity_buffer: Distance threshold for proximity

    Returns:
        True if rectangle is inside table or within proximity buffer
    """
    table_rect = table.bounding_rect

    # Check if rectangle is inside table
    if table.contains_rect(rect):
        return True

    # Check if rectangle is within proximity buffer of table
    expanded_table_rect = pymupdf.Rect(
        table_rect.x0 - proximity_buffer,
        table_rect.y0 - proximity_buffer,
        table_rect.x1 + proximity_buffer,
        table_rect.y1 + proximity_buffer
    )

    # Check for any overlap with expanded table area
    return not (
        rect.x1 < expanded_table_rect.x0 or
        rect.x0 > expanded_table_rect.x1 or
        rect.y1 < expanded_table_rect.y0 or
        rect.y0 > expanded_table_rect.y1
    )
