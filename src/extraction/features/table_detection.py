"""This module contains functionalities to detect table like structures"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Point, Line

from utils.file_utils import read_params

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
    page_height: float = None
) -> List[TableStructure]:
    """Detect large table structures on a page.

    Args:
        geometric_lines: List of detected geometric lines
        page_width: Page width 
        page_height: Page height

    Returns:
        List containing at most one large table structure
    """
    config = read_params('table_detection_params.yml')

    # Need sufficient lines for a meaningful table
    if len(geometric_lines) < config.get('min_total_lines'):
        logger.debug(f"Insufficient lines ({len(geometric_lines)}) for table detection")
        return []

    # Filter and classify lines
    filtered_lines = _filter_significant_lines(geometric_lines, config, page_width, page_height)
    horizontal_lines, vertical_lines = _separate_by_orientation(filtered_lines, config)

    logger.debug(f"Found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")

    # Need substantial line structure for a table
    if len(horizontal_lines) < 4 or len(vertical_lines) < 2:
        logger.debug("Insufficient line structure for table detection")
        return []

    # Find the dominant table structure
    table_candidate = _find_dominant_table_structure(
        horizontal_lines, vertical_lines, config, page_width, page_height
    )

    if table_candidate and table_candidate.confidence >= config.get('min_confidence'):
        logger.info(f"Detected large table structure (confidence: {table_candidate.confidence:.3f})")
        return [table_candidate]

    logger.debug("No significant table structure found")
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
        elif abs(angle - 90) <= angle_tolerance:
            vertical_lines.append(line)

    return horizontal_lines, vertical_lines


def _find_dominant_table_structure(
    horizontal_lines: List[Line],
    vertical_lines: List[Line], 
    config: dict,
    page_width: float = None,
    page_height: float = None
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
    line_density = len(table_horizontal_lines + table_vertical_lines) / (area / 10000)

    # Calculate confidence based on structure quality
    confidence = _calculate_structure_confidence(
        refined_rect, table_horizontal_lines, table_vertical_lines,
        line_density, config, page_width, page_height
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
        main_verticals = v_sorted[:min(6, len(v_sorted))]  # Top 6 longest

        # Get x-coordinates of main vertical lines
        x_positions = []
        for line in main_verticals:
            x_pos = (line.start.x + line.end.x) / 2
            x_positions.append(x_pos)

        if x_positions:
            refined_min_x = min(x_positions) - 10  # Small buffer
            refined_max_x = max(x_positions) + 10
        else:
            refined_min_x = initial_rect.x0
            refined_max_x = initial_rect.x1
    else:
        refined_min_x = initial_rect.x0
        refined_max_x = initial_rect.x1

    # Find the main horizontal boundaries
    if len(horizontal_lines) >= 3:
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
    page_height: float = None
) -> float:
    """Calculate confidence score for the table structure."""

    area = rect.width * rect.height

    # Size score - larger tables are more likely to be significant
    if page_width and page_height:
        page_area = page_width * page_height
        area_ratio = area / page_area
        size_score = min(1.0, area_ratio / config.get('min_table_area_ratio'))
    else:
        # Fallback based on absolute size
        size_score = min(1.0, area / 50000)

    # Line structure score - more lines indicate better structure
    total_lines = len(h_lines) + len(v_lines)
    line_score = min(1.0, total_lines / 15.0)  # 15 lines = perfect score

    # Balance score - good tables have both horizontal and vertical lines
    h_count, v_count = len(h_lines), len(v_lines)
    if h_count > 0 and v_count > 0:
        balance_score = min(h_count, v_count) / max(h_count, v_count)
    else:
        balance_score = 0.0

    # Line span score - lines should span significant portions
    h_span_score = _calculate_span_score(h_lines, rect.width, is_horizontal=True)
    v_span_score = _calculate_span_score(v_lines, rect.height, is_horizontal=False)
    span_score = (h_span_score + v_span_score) / 2

    # Weighted combination
    total_confidence = (
        size_score * 0.35 +      # Size is important for "large" structures  
        line_score * 0.25 +      # Line count matters
        balance_score * 0.20 +   # Need both h and v lines
        span_score * 0.20        # Lines should span the structure
    )

    return min(1.0, total_confidence)


def _calculate_span_score(lines: List[Line], dimension: float, is_horizontal: bool) -> float:
    """Calculate how well lines span across the structure."""
    if not lines or dimension <= 0:
        return 0.0

    spans = []
    for line in lines:
        if is_horizontal:
            line_span = abs(line.end.x - line.start.x)
        else:
            line_span = abs(line.end.y - line.start.y)

        span_ratio = line_span / dimension
        spans.append(min(1.0, span_ratio))

    # Average span ratio of all lines
    return sum(spans) / len(spans) if spans else 0.0


# Compatibility function for existing code
def filter_extraction_pairs_by_tables(
    material_description_pairs: List,
    table_structures: List[TableStructure],
) -> List:
    """Filter material description pairs by table structures."""
    if not table_structures:
        return material_description_pairs

    return material_description_pairs
