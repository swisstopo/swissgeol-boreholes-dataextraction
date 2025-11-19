"""This module contains the template matching code.

The code in this file aims to extract groundwater information based on the location where
the groundwater illustration was found in the document of interest.
"""

import logging
import re

import pymupdf

from extraction.features.groundwater.utility import extract_date
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params

logger = logging.getLogger(__name__)

merging_params = read_params("line_detection_params.yml")["line_merging_params"]


def get_text_lines_near_symbol(
    lines: list[TextLine],
    geo_line: Line,
) -> list[TextLine]:
    """Extracts the text lines that are close to a groundwater symbol.

    Args:
        lines (list[TextLine]): The list of text lines to search.
        geo_line (Line): the upper geometric line of a groundwater symbol.

    Returns:
        list[TextLine]: The list of text lines near groundwater symbol.
    """
    x0, x1, y, length = geo_line.start.x, geo_line.end.x, geo_line.start.y, geo_line.length
    search_rect = pymupdf.Rect(x0 - 0.5 * length, y - length, x1 + 0.5 * length, y + length)
    feature_lines = [line for line in lines if line.rect.intersects(search_rect)]
    # sort, to first treat the lines close to the symbol
    feature_lines.sort(
        key=lambda line: ((line.rect.x0 + line.rect.x1) / 2 - (x0 + x1) / 2) ** 2
        + ((line.rect.y0 + line.rect.y1) / 2 - y) ** 2
    )
    return feature_lines


def get_groundwater_symbol_upper_lines(
    lines: list[TextLine],
    geometric_lines: list[Line],
) -> list[Line]:
    """Extracts the upper geometric lines that belong to a groundwater symbol.

    The function looks for either depth values (numbers) or dates, and checks if they are associated with nearby
    horizontal lines that could indicate groundwater symbols. The function filters out depths that are stratigraphy
    informations. Typically, groundwater symbols look like that:

            ------------------------
                -----------------

    Args:
        lines (list[TextLine]): The list of text lines to search.
        geometric_lines (list[Line]): The list of geometric lines to consider.


    Returns:
        list[Line]: The list containing the upper geometric line of each groundwater symbol.
    """
    pair_bigger_lines: list[Line] = []
    for text_line in lines:
        depth_pattern = re.compile(r"^-?([0-9]+(?:[\.,][0-9]+)?)")
        match = depth_pattern.match(text_line.text)
        if not match:
            extracted_date, _ = extract_date(text_line.text)
            if extracted_date is None:
                continue

        def is_near_entry(line: Line, x0: float, y0: float, x1: float, y1: float):
            dx, dy = x1 - x0, y1 - y0
            return any(
                x0 - dx <= line_x <= x1 + dx and y0 - dy <= line_y <= y1 + dy
                for line_x, line_y in (line.start.as_numpy, line.end.as_numpy)
            )

        text_width = text_line.rect.x1 - text_line.rect.x0
        horizontals_near_entry = [
            g_line
            for g_line in geometric_lines
            if g_line.is_horizontal(merging_params["horizontal_slope_tolerance"])
            and is_near_entry(g_line, *text_line.rect)
            and g_line.length < 3 * text_width
        ]
        pairs = get_valid_pairs(horizontals_near_entry, text_line, lines)
        if not pairs:
            continue

        pair_bigger_lines.extend([pair[0] for pair in pairs])

    return pair_bigger_lines


def is_valid_pair(l_top: Line, l_bot: Line, hit_line: TextLine, text_lines: list[TextLine]):
    """Check if the given line pair represents a groundwater symbol based on various heuristics.

    Args:
        l_top (Line): The top line.
        l_bot (Line): The bottom line.
        hit_line (TextLine): The TextLine that contains the first detected element.
        text_lines (list[TextLine]): The list of all text lines.

    Returns:
        bool: True if the pair is valid, False otherwise.
    """
    avg_text_size = sum(line.rect.height for line in text_lines) / len(text_lines)
    rel_size_ok = l_top.length * 0.8 > l_bot.length > l_top.length / 4  # bottom one is smaler, but not too small
    global_size_ok = avg_text_size < l_top.length < 3 * avg_text_size  # size comparable to avg text height
    pos_ok = l_top.start.x < l_bot.start.x and l_bot.end.x < l_top.end.x  # the top one fully "spans" the bottom one
    pos_to_bb_ok = hit_line.rect.x0 < l_top.end.x and l_top.start.x < hit_line.rect.x1  # longest line overlaps the bb
    gap_ok = l_bot.start.y - l_top.start.y < l_bot.length / 3  # gap is small, relative to the bottom line
    no_bb_in_gap = not any(
        line.rect.intersects(pymupdf.Rect(l_top.start.x, l_top.start.y, l_top.end.x, l_bot.start.y))
        for line in text_lines
    )  # no text line sits in between the two lines

    return rel_size_ok and global_size_ok and pos_ok and pos_to_bb_ok and gap_ok and no_bb_in_gap


def get_valid_pairs(lines: list[Line], line: TextLine, text_lines: list[TextLine]) -> list[tuple[Line, Line]]:
    """Get all valid pairs of lines that could represent a groundwater symbol.

    Args:
        lines (list[Line]): The list of lines to consider.
        line (TextLine): The TextLine that contains the first detected element.
        text_lines (list[TextLine]): The list of all text lines.

    Returns:
        list[tuple[Line, Line]]: A list of valid line pairs.
    """
    pairs = []

    lines.sort(key=lambda line: line.start.y)  # top down
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines):
            if j <= i:
                continue
            if is_valid_pair(l1, l2, line, text_lines):
                pairs.append((l1, l2))
    return pairs
