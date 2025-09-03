"""Test suite for the interval module."""

import pymupdf
import pytest

from extraction.features.stratigraphy.interval.a_to_b_interval_extractor import AToBIntervalExtractor
from extraction.features.utils.text.textline import TextLine, TextWord

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
    interval, _ = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval.start.value == 1.70, "The start value of the interval should be 1.70m"
    assert interval.start.rect == pymupdf.Rect(2, 0, 3, 1), "The start rect of the interval should be correct"
    assert interval.end.value == 5.80, "The start value of the interval should be 5.80m"
    assert interval.end.rect == pymupdf.Rect(4, 0, 5, 1), "The end rect of the interval should be correct"

    words = [
        TextWord(pymupdf.Rect(0, 0, 2, 1), "COLLUVIONS:", page=1),
        TextWord(pymupdf.Rect(2, 0, 6, 1), "1,70-5,80m", page=1),
    ]
    interval, _ = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval.start.value == 1.70, "The start value of the interval should be 1.70m"
    assert interval.start.rect == pymupdf.Rect(2, 0, 6, 1), "The start rect of the interval should be correct"
    assert interval.end.value == 5.80, "The start value of the interval should be 5.80m"
    assert interval.end.rect == pymupdf.Rect(2, 0, 6, 1), "The end rect of the interval should be correct"

    words = [
        TextWord(pymupdf.Rect(0, 0, 2, 1), "COLLUVIONS:", page=1),
        TextWord(pymupdf.Rect(2, 0, 6, 1), "1,70-5,80m", page=1),
    ]
    interval, _ = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=True)
    assert interval is None, "With require_start_of_string=True, matches that are not at the start are not allowed."

    words = [
        TextWord(pymupdf.Rect(0, 0, 4, 1), "1,70-5,80m:", page=1),
        TextWord(pymupdf.Rect(4, 0, 6, 1), "COLLUVIONS", page=1),
    ]
    interval, _ = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval is not None, "With require_start_of_string=False, matches that are at the start are allowed."

    words = [
        TextWord(pymupdf.Rect(0, 0, 4, 1), "Argile", page=1),
        TextWord(pymupdf.Rect(4, 0, 6, 1), "dès 4m.", page=1),
    ]
    interval, _ = AToBIntervalExtractor.from_text(TextLine(words), require_start_of_string=False)
    assert interval is not None and interval.end is None, "Open-ended intervals are allowed."


@pytest.fixture
def create_line():
    """Fixture providing a function to create TextBlock with given text and position."""

    def _create_line(text: str) -> TextLine:
        words = [TextWord(pymupdf.Rect(i, 0, i + 1, 1), w, page=1) for i, w in enumerate(text.split())]
        return TextLine(words)

    return _create_line


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("5 - 7.1 m Clay and silt", "Clay and silt"),
        ("5,0 - 7,1 m Clay and silt", "Clay and silt"),
        ("5.0 - 7.1 m Clay and silt", "Clay and silt"),
        ("5,00 - 7,10 m Clay and silt", "Clay and silt"),
        ("5.00 - 7.10 m Clay and silt", "Clay and silt"),
        ("5.000 - 7.100 m Clay and silt", "Clay and silt"),
        ("5,000 - 7,100 m Clay and silt", "Clay and silt"),
    ],
)
def test_clean_block_number_formats(create_line, input_text, expected):
    """Test clean_block with different number formats."""
    _, line_stripped = AToBIntervalExtractor.from_text(create_line(input_text), require_start_of_string=False)
    assert line_stripped.text == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("5-7.1m Clay and silt", "Clay and silt"),
        ("5 -7.1m Clay and silt", "Clay and silt"),
        ("5- 7.1m Clay and silt", "Clay and silt"),
        ("5  -  7.1  m  Clay and silt", "Clay and silt"),
        ("   5-7.1m   Clay and silt", "Clay and silt"),
    ],
)
def test_clean_block_spacing_patterns(create_line, input_text, expected):
    """Test clean_block with different spacing patterns."""
    _, line_stripped = AToBIntervalExtractor.from_text(create_line(input_text), require_start_of_string=False)
    assert line_stripped.text == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("5-7.1m: Clay and silt", "Clay and silt"),
        ("5-7.1m; Clay and silt", "Clay and silt"),
        ("5-7.1m. Clay and silt", "Clay and silt"),
    ],
)
def test_clean_block_punctuation(create_line, input_text, expected):
    """Test clean_block with different punctuation marks."""
    _, line_stripped = AToBIntervalExtractor.from_text(create_line(input_text), require_start_of_string=False)
    assert line_stripped.text == expected


@pytest.mark.parametrize(
    "input_text",
    [
        "Clay and silt",
        "Layer 5: Clay and silt",
        "From the surface: Clay and silt",
    ],
)
def test_clean_block_no_match(create_line, input_text):
    """Test clean_block when there's no depth information to clean."""
    _, line_stripped = AToBIntervalExtractor.from_text(create_line(input_text), require_start_of_string=False)
    assert line_stripped.text == input_text


def test_clean_block_empty_result(create_line):
    """Test clean_block when cleaning would result in an empty line."""
    _, line_stripped = AToBIntervalExtractor.from_text(create_line("5-7.1m"), require_start_of_string=False)
    assert len(line_stripped.words) == 0
