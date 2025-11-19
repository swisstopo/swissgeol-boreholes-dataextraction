"""Tests for groundwater symbol detection."""

import pymupdf
import pytest

from extraction.features.groundwater.groundwater_symbol_detection import is_valid_pair
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.text.textline import TextLine, TextWord


@pytest.mark.parametrize(
    "l_top,l_bot,hit_line,expected,test_name",
    [
        # Valid case
        (
            # Case 1: Perfect groundwater symbol - longer top line, shorter bottom line, proper spacing
            Line(Point(100, 100), Point(120, 100)),  # top line
            Line(Point(103, 104), Point(117, 104)),  # bottom line - shorter and below
            TextLine([TextWord(pymupdf.Rect(110, 90, 130, 99), "12.5", 0)]),  # text overlaps with top line
            True,
            "valid_perfect_symbol",
        ),
        # Invalid cases
        (
            # Case 2: bottom line longer than 80% of top line
            Line(Point(100, 100), Point(120, 100)),
            Line(Point(101, 104), Point(119, 104)),
            TextLine([TextWord(pymupdf.Rect(110, 90, 130, 99), "12.5", 0)]),
            False,
            "invalid_bottom_line_too_long",
        ),
        (
            # Case 3: gap is too big
            Line(Point(100, 100), Point(120, 100)),
            Line(Point(103, 106), Point(117, 106)),
            TextLine([TextWord(pymupdf.Rect(110, 90, 130, 99), "12.5", 0)]),
            False,
            "invalid_gap_too_big",
        ),
        (
            # Case 4: symbol too big compared to avg text size
            Line(Point(100, 100), Point(140, 100)),
            Line(Point(108, 115), Point(132, 115)),
            TextLine([TextWord(pymupdf.Rect(110, 90, 130, 99), "12.5", 0)]),
            False,
            "invalid_symbol_too_big",
        ),
        (
            # Case 5: text not overlapping the top line x-values
            Line(Point(100, 100), Point(120, 100)),
            Line(Point(103, 104), Point(117, 104)),
            TextLine([TextWord(pymupdf.Rect(121, 90, 140, 99), "12.5", 0)]),
            False,
            "invalid_textline_not_close_enough",
        ),
        (
            # Case 6: text sit in between the lines
            Line(Point(100, 100), Point(120, 100)),
            Line(Point(103, 104), Point(117, 104)),
            TextLine([TextWord(pymupdf.Rect(110, 100, 130, 110), "12.5", 0)]),
            False,
            "invalid_textline_inside_gap",
        ),
    ],
)
def test_is_valid_pair(l_top, l_bot, hit_line, expected, test_name):
    """Test is_valid_pair function with various line configurations.

    Args:
        l_top: Top Line object
        l_bot: Bottom Line object
        hit_line: TextLine object representing the detected text
        expected: Expected boolean result
        test_name: Name of the test case for better error reporting
    """
    # Create a list of text lines with consistent average height
    text_lines = [
        TextLine([TextWord(pymupdf.Rect(50, 50, 70, 60), "Sample", 0)]),
        TextLine([TextWord(pymupdf.Rect(80, 50, 100, 60), "Text", 0)]),
        hit_line,
    ]

    result = is_valid_pair(l_top, l_bot, hit_line, text_lines)
    assert result == expected, f"Test case '{test_name}' failed: expected {expected}, got {result}"
