"""Simplified table detection for borehole PDF documents.

Detects rectangular areas with contiguous horizontal and vertical lines.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

logger = logging.getLogger(__name__)

@dataclass
class TableStructure:
    """Table structure representation."""
    bounding_rect: pymupdf.Rect
    horizontal_lines: List[Line]
    vertical_lines: List[Line]
    confidence: float

    def contains_rect(self, rect: pymupdf.Rect) -> bool:
        """Check if a rectangle is within this table structure."""
        return (self.bounding_rect.x0 <= rect.x0 and rect.x1 <= self.bounding_rect.x1 and
                self.bounding_rect.y0 <= rect.y0 and rect.y1 <= self.bounding_rect.y1)

    def overlaps_with(self, other: 'TableStructure', threshold: float = 0.4) -> bool:
        """Check if this table overlaps significantly with another."""
        intersection = self.bounding_rect & other.bounding_rect
        if intersection.is_empty:
            return False

        overlap_area = intersection.width * intersection.height
        min_area = min(self.bounding_rect.width * self.bounding_rect.height,
                      other.bounding_rect.width * other.bounding_rect.height)

        return overlap_area / min_area > threshold


def detect_table_structures(
    geometric_lines: List[Line],
    text_lines: List[TextLine] = None,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None
) -> List[TableStructure]:
    """Detect rectangular table structures from geometric lines.

    Args:
        geometric_lines: List of detected geometric lines
        text_lines: Text lines (optional, for basic validation)
        page_width: Page width (optional, for boundary detection)
        page_height: Page height (optional, for boundary detection)
        config: Configuration dictionary (optional, loads from file if not provided)

    Returns:
        List of detected table structures
    """
    # Load configuration
    config = read_params('table_detection_params.yml')

    if len(geometric_lines) < 4:  # Need at least 4 lines for a rectangle
        return []

    # Skip table detection if too many lines (performance killer)
    if len(geometric_lines) > config['max_lines_threshold']:
        logger.warning(f"Skipping table detection: too many geometric lines ({len(geometric_lines)})")
        return []

    # Filter out page boundary lines if page dimensions are provided !CURENTLY NOT USED
    # if page_width and page_height and config['page_boundary_detection']['enabled']:
    #     geometric_lines = _filter_page_boundary_lines(geometric_lines, page_width, page_height, config)

    # Separate and filter lines
    h_lines, v_lines = _separate_lines(geometric_lines, config['angle_tolerance'])

    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # Filter lines by length
    h_lines_filtered = _filter_lines_by_length(
        h_lines, config['min_table_width'] * config['line_length_ratios']['horizontal_min_ratio'], "horizontal"
    )
    v_lines_filtered = _filter_lines_by_length(
        v_lines, config['min_table_height'] * config['line_length_ratios']['vertical_min_ratio'], "vertical"
    )

    logger.debug(f"Line filtering: {len(h_lines)} -> {len(h_lines_filtered)} horizontal, "
                f"{len(v_lines)} -> {len(v_lines_filtered)} vertical lines")

    # Find table candidates
    candidates = _find_table_candidates(
        h_lines_filtered, v_lines_filtered, 
        config['min_table_width'], config['min_table_height'], config
    )

    # Create table structures with confidence scoring
    tables = []
    for candidate in candidates:
        confidence = _calculate_confidence(candidate, text_lines, config)
        if confidence > config['min_confidence_threshold']:
            tables.append(TableStructure(
                bounding_rect=candidate['rect'],
                horizontal_lines=candidate['h_lines'],
                vertical_lines=candidate['v_lines'],
                confidence=confidence
            ))

    # Remove overlapping tables and sort by confidence
    tables = _remove_overlaps(tables, config['overlap_removal_threshold'])
    tables.sort(key=lambda t: t.confidence, reverse=True)

    logger.info(f"Detected {len(tables)} table structures")
    return tables


# def _filter_page_boundary_lines(
#     geometric_lines: List[Line],
#     page_width: float,
#     page_height: float,
#     config: dict
# ) -> List[Line]:
#     """Filter out lines that form the page boundary to avoid detecting full page as table."""
#     boundary_tolerance = config['page_boundary_detection']['boundary_tolerance']
#     filtered_lines = []

#     for line in geometric_lines:
#         is_boundary = False

#         # Check if line is near page edges
#         # Top boundary
#         if (abs(line.start.y) < boundary_tolerance and abs(line.end.y) < boundary_tolerance and
#             abs(line.start.x - line.end.x) > page_width * config['page_boundary_detection']['min_boundary_coverage']):
#             is_boundary = True

#         # Bottom boundary  
#         elif (abs(line.start.y - page_height) < boundary_tolerance and 
#               abs(line.end.y - page_height) < boundary_tolerance and
#               abs(line.start.x - line.end.x) > page_width * config['page_boundary_detection']['min_boundary_coverage']):
#             is_boundary = True

#         # Left boundary
#         elif (abs(line.start.x) < boundary_tolerance and abs(line.end.x) < boundary_tolerance and
#               abs(line.start.y - line.end.y) > page_height * config['page_boundary_detection']['min_boundary_coverage']):
#             is_boundary = True

#         # Right boundary
#         elif (abs(line.start.x - page_width) < boundary_tolerance and 
#               abs(line.end.x - page_width) < boundary_tolerance and
#               abs(line.start.y - line.end.y) > page_height * config['page_boundary_detection']['min_boundary_coverage']):
#             is_boundary = True

#         if not is_boundary:
#             filtered_lines.append(line)

#     if len(filtered_lines) < len(geometric_lines):
#         logger.debug(f"Filtered {len(geometric_lines) - len(filtered_lines)} page boundary lines")

#     return filtered_lines


def _filter_lines_by_length(lines: List[Line], min_length: float, line_type: str) -> List[Line]:
    """Filter lines by minimum length to focus on meaningful table structure lines.

    This is a critical performance optimization that removes short decorative lines
    that cannot possibly form table structures. Borehole documents typically have
    many small decorative lines that dramatically slow down table detection.

    Args:
        lines: List of lines to filter
        min_length: Minimum line length to keep

    Returns:
        Filtered list of lines that meet the minimum length requirement
    """
    filtered_lines = []

    for line in lines:
        # Calculate line length using Euclidean distance
        length = ((line.end.x - line.start.x) ** 2 + (line.end.y - line.start.y) ** 2) ** 0.5

        # For vertical lines in borehole profiles, prioritize longer lines more aggressively
        if line_type == "vertical" and abs(line.angle - 90) <= 5:
            # Be more permissive with vertical lines - they define column boundaries
            effective_min_length = min_length * 0.3
        elif line_type == "horizontal" and (abs(line.angle) <= 5 or abs(line.angle) >= 175):
            # Be more strict with horizontal lines - only keep substantial ones
            effective_min_length = min_length * 0.5
        else:
            effective_min_length = min_length

        if length >= effective_min_length:
            filtered_lines.append(line)

    return filtered_lines


def _separate_lines(geometric_lines: List[Line], angle_tolerance: float = 5.0) -> tuple[List[Line], List[Line]]:
    """Separate lines into horizontal and vertical categories."""
    horizontal_lines = []
    vertical_lines = []

    for line in geometric_lines:
        angle = abs(line.angle)
        if angle <= angle_tolerance or angle >= (180 - angle_tolerance):  # Horizontal
            horizontal_lines.append(line)
        elif abs(angle - 90) <= angle_tolerance:  # Vertical
            vertical_lines.append(line)

    return horizontal_lines, vertical_lines


def _find_table_candidates(
    h_lines: List[Line],
    v_lines: List[Line],
    min_width: float,
    min_height: float,
    config: dict
) -> List[dict]:
    """Find rectangular table candidates from line intersections."""
    candidates = []

    # Sort vertical lines by length (prioritize long column boundaries)
    v_lines_sorted = sorted(v_lines,
                           key=lambda line: ((line.end.x - line.start.x) ** 2 + (line.end.y - line.start.y) ** 2) ** 0.5,
                           reverse=True)

    max_candidates = config['candidate_validation']['max_candidates_to_check']
    min_h_lines = config['candidate_validation']['min_horizontal_lines']
    tall_ratio = config['candidate_validation']['tall_structure_height_ratio']

    # Start with the longest vertical lines as column boundaries
    for i, left_v in enumerate(v_lines_sorted[:max_candidates]):
        for right_v in v_lines_sorted[i+1:]:
            width = abs(right_v.start.x - left_v.start.x)
            if width < min_width:
                continue

            # Find the extent of these vertical lines to determine table height
            min_y = min(left_v.start.y, left_v.end.y, right_v.start.y, right_v.end.y)
            max_y = max(left_v.start.y, left_v.end.y, right_v.start.y, right_v.end.y)
            height = max_y - min_y

            if height < min_height:
                continue

            # Create potential table rectangle
            rect = pymupdf.Rect(
                min(left_v.start.x, left_v.end.x),
                min_y,
                max(right_v.start.x, right_v.end.x),
                max_y
            )

            # Find horizontal lines within this rectangle
            rect_h_lines = [line for line in h_lines if _line_in_rect(line, rect)]
            rect_v_lines = [left_v, right_v] + [line for line in v_lines if _line_in_rect(line, rect) and line not in [left_v, right_v]]

            # Check if this qualifies as a table candidate
            if len(rect_h_lines) >= min_h_lines or height > min_height * tall_ratio:
                candidates.append({
                    'rect': rect,
                    'h_lines': rect_h_lines,
                    'v_lines': rect_v_lines,
                    'aspect_ratio': height / width  # Track tall structures
                })

    return candidates


def _line_in_rect(line: Line, rect: pymupdf.Rect, tolerance: float = 5.0) -> bool:
    """Check if a line is within a rectangle."""
    expanded_rect = pymupdf.Rect(
        rect.x0 - tolerance, rect.y0 - tolerance,
        rect.x1 + tolerance, rect.y1 + tolerance
    )

    # Check if line endpoints are within the expanded rectangle
    start_in = (expanded_rect.x0 <= line.start.x <= expanded_rect.x1 and
                expanded_rect.y0 <= line.start.y <= expanded_rect.y1)
    end_in = (expanded_rect.x0 <= line.end.x <= expanded_rect.x1 and
              expanded_rect.y0 <= line.end.y <= expanded_rect.y1)

    return start_in or end_in


def _calculate_confidence(candidate: dict, text_lines: List[TextLine] = None, config: dict = None) -> float:
    """Calculate confidence score based on line density and text presence."""
    config = read_params('table_detection_params.yml')

    rect = candidate['rect']
    h_count = len(candidate['h_lines'])
    v_count = len(candidate['v_lines'])

    # Base confidence from line count
    total_lines = h_count + v_count
    confidence = min(0.5, total_lines * config['confidence_scoring']['base_line_weight'])

    # Bonus for having both horizontal and vertical lines
    if h_count >= 2 and v_count >= 2:
        confidence += config['confidence_scoring']['both_directions_bonus']

    # Bonus for text content within the rectangle
    if text_lines:
        text_within = [line for line in text_lines if rect.intersects(line.rect)]
        if text_within:
            text_bonus = min(
                config['confidence_scoring']['max_text_bonus'], 
                len(text_within) * config['confidence_scoring']['text_presence_weight']
                )
            confidence += text_bonus

    return min(1.0, confidence)


def _remove_overlaps(tables: List[TableStructure], threshold: float = 0.4) -> List[TableStructure]:
    """Remove overlapping tables, keeping highest confidence ones."""
    if len(tables) <= 1:
        return tables

    # Sort by area and confidence
    tables.sort(key=lambda t: (t.bounding_rect.width * t.bounding_rect.height, t.confidence), reverse=True)
    filtered = []

    for table in tables:
        # Check overlap with already accepted tables
        overlaps = False
        for existing in filtered:
            if table.overlaps_with(existing, threshold):
                overlaps = True
                break

        if not overlaps:
            filtered.append(table)

    return filtered



## Initial implementation needs to be changed
def filter_extraction_pairs_by_tables(
    material_description_pairs: List,
    table_structures: List[TableStructure],
    max_pairs_per_table: int = 1
) -> List:
    """Filter material description pairs to only include those within detected table structures.

    This enforces the constraint that each table structure contains at most a limited number of boreholes,
    helping to eliminate duplicates

    Args:
        material_description_pairs: List of MaterialDescriptionRectWithSidebar pairs
        table_structures: List of detected table structures
        max_pairs_per_table: Maximum number of pairs to accept per table (default: 1)

    Returns:
        Filtered list of pairs, with at most max_pairs_per_table per table structure
    """
    if not table_structures:
        # No table structures detected, return original pairs
        logger.debug("No table structures detected, returning all pairs")
        return material_description_pairs

    # If we have many table structures, allow 1 borehole per table 
    # If we have few tables but many pairs, this indicates extraction noise 
    if len(table_structures) > 2 and len(material_description_pairs) > len(table_structures):
        logger.info(f"Multiple table structures ({len(table_structures)}) detected - allowing 1 borehole per table "
                   f"for potential multi-borehole page")
        max_pairs_per_table = 1  # Keep strict: 1 borehole per table structure
    elif len(table_structures) == 1 and len(material_description_pairs) > 3:
        logger.warning(f"Single table with many pairs ({len(material_description_pairs)}) detected - "
                      f"likely extraction noise/duplicates, applying strict filtering")
        max_pairs_per_table = 1  # Stay strict to filter out noise

    filtered_pairs = []
    table_pair_counts = {}  # Track how many pairs we've accepted per table

    # Sort pairs by match score (highest first)
    sorted_pairs = sorted(material_description_pairs, key=lambda p: p.score_match, reverse=True)

    # First pass: assign pairs to their best-matching tables
    pairs_outside_tables = []

    for pair in sorted_pairs:
        # Find which table(s) contain this pair
        containing_tables = []

        for i, table in enumerate(table_structures):
            # Check if the material description rectangle is within this table
            if table.contains_rect(pair.material_description_rect):
                containing_tables.append((i, table))

        if containing_tables:
            # Choose the table with highest confidence that still has capacity
            best_table = None
            for table_idx, table in sorted(containing_tables, key=lambda x: x[1].confidence, reverse=True):
                if table_pair_counts.get(table_idx, 0) < max_pairs_per_table:
                    best_table = table_idx
                    break

            if best_table is not None:
                filtered_pairs.append(pair)
                table_pair_counts[best_table] = table_pair_counts.get(best_table, 0) + 1
                logger.debug(f"Accepted pair in table {best_table} with score {pair.score_match:.2f} "
                           f"(count: {table_pair_counts[best_table]}/{max_pairs_per_table})")
            else:
                logger.debug(f"All matching tables at capacity for pair with score {pair.score_match:.2f}")
        else:
            # Pair is not within any table
            pairs_outside_tables.append(pair)

    # Second pass: if we filtered too aggressively and have very few pairs,
    # consider adding back high-quality pairs that are outside tables
    if len(filtered_pairs) < len(material_description_pairs) * 0.3:  # If we filtered more than 70%
        logger.info(f"Table filtering removed {len(material_description_pairs) - len(filtered_pairs)} pairs "
                   f"({100 * (1 - len(filtered_pairs)/len(material_description_pairs)):.1f}%) - "
                   f"considering pairs outside tables")

        # Add back high-scoring pairs that are outside tables
        high_quality_outside = [p for p in pairs_outside_tables if p.score_match > 0.7]
        if high_quality_outside:
            filtered_pairs.extend(high_quality_outside[:2])  # Add at most 2 high-quality outside pairs
            logger.info(f"Added back {len(high_quality_outside[:2])} high-quality pairs outside tables")

    logger.info(f"Table-based filtering: {len(material_description_pairs)} -> {len(filtered_pairs)} pairs "
               f"({len(pairs_outside_tables)} were outside tables)")
    return filtered_pairs
