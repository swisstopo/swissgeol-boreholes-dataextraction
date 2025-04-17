"""Test suite for the interval module."""

import pymupdf
from borehole_extraction.extraction.stratigraphy.depth.a_to_b_interval_extractor import AToBIntervalExtractor
from borehole_extraction.extraction.util_extraction.lines.line import TextLine, TextWord

# TODO: add tests for AAboveBInterval.matching_blocks


def test_atobintervalextractor_fromtext():  # noqa: D103
    """Test the from_text class-method of the AToBIntervalExtractor class."""
    words = [
        TextWord(pymupdf.Rect(0, 0, 2, 1), "COLLUVIONS:", page=1),
        TextWord(pymupdf.Rect(2, 0, 3, 1), "1,70", page=1),
        TextWord(pymupdf.Rect(3, 0, 4, 1), "-", page=1),
        TextWord(pymupdf.Rect(4, 0, 5, 1), "5,80", page=1),
        TextWord(pymupdf.Rect(5, 0, 6, 1), "m", page=1),
    ]
    interval = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval.start.value == 1.70, "The start value of the interval should be 1.70m"
    assert interval.start.rect == pymupdf.Rect(2, 0, 3, 1), "The start rect of the interval should be correct"
    assert interval.end.value == 5.80, "The start value of the interval should be 5.80m"
    assert interval.end.rect == pymupdf.Rect(4, 0, 5, 1), "The end rect of the interval should be correct"

    words = [
        TextWord(pymupdf.Rect(0, 0, 2, 1), "COLLUVIONS:", page=1),
        TextWord(pymupdf.Rect(2, 0, 6, 1), "1,70-5,80m", page=1),
    ]
    interval = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval.start.value == 1.70, "The start value of the interval should be 1.70m"
    assert interval.start.rect == pymupdf.Rect(2, 0, 6, 1), "The start rect of the interval should be correct"
    assert interval.end.value == 5.80, "The start value of the interval should be 5.80m"
    assert interval.end.rect == pymupdf.Rect(2, 0, 6, 1), "The end rect of the interval should be correct"

    words = [
        TextWord(pymupdf.Rect(0, 0, 2, 1), "COLLUVIONS:", page=1),
        TextWord(pymupdf.Rect(2, 0, 6, 1), "1,70-5,80m", page=1),
    ]
    interval = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=True)
    assert interval is None, "With require_start_of_string=True, matches that are not at the start are not allowed."

    words = [
        TextWord(pymupdf.Rect(0, 0, 4, 1), "1,70-5,80m:", page=1),
        TextWord(pymupdf.Rect(4, 0, 6, 1), "COLLUVIONS", page=1),
    ]
    interval = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval is not None, "With require_start_of_string=False, matches that are at the start are allowed."
