"""Test suite for the IntervalBlockPair class."""

import pymupdf
import pytest

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.interval import AToBInterval, IntervalBlockPair
from extraction.features.utils.text.textblock import TextBlock
from extraction.features.utils.text.textline import TextLine, TextWord


@pytest.fixture
def interval():
    """Create a default interval fixture with depth 5-7.1m."""
    start = DepthColumnEntry(value=5.0, rect=pymupdf.Rect(0, 0, 1, 1))
    end = DepthColumnEntry(value=7.1, rect=pymupdf.Rect(1, 0, 2, 1))
    return AToBInterval(start=start, end=end)


@pytest.fixture
def create_block():
    """Fixture providing a function to create TextBlock with given text and position."""

    def _create_block(text: str) -> TextBlock:
        words = [TextWord(pymupdf.Rect(i, 0, i + 1, 1), w, page=1) for i, w in enumerate(text.split())]
        return TextBlock([TextLine(words)])

    return _create_block


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
def test_clean_block_number_formats(interval, create_block, input_text, expected):
    """Test clean_block with different number formats."""
    pair = IntervalBlockPair(interval, create_block(input_text))
    new_block = pair.get_clean_block()
    assert new_block.lines[0].text == expected


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
def test_clean_block_spacing_patterns(interval, create_block, input_text, expected):
    """Test clean_block with different spacing patterns."""
    pair = IntervalBlockPair(interval, create_block(input_text))
    new_block = pair.get_clean_block()
    assert new_block.lines[0].text == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("5-7.1m: Clay and silt", "Clay and silt"),
        ("5-7.1m; Clay and silt", "Clay and silt"),
        ("5-7.1m. Clay and silt", "Clay and silt"),
    ],
)
def test_clean_block_punctuation(interval, create_block, input_text, expected):
    """Test clean_block with different punctuation marks."""
    pair = IntervalBlockPair(interval, create_block(input_text))
    new_block = pair.get_clean_block()
    assert new_block.lines[0].text == expected


@pytest.mark.parametrize(
    "input_text",
    [
        "Clay and silt",
        "Layer 5: Clay and silt",
        "From the surface: Clay and silt",
    ],
)
def test_clean_block_no_match(interval, create_block, input_text):
    """Test clean_block when there's no depth information to clean."""
    pair = IntervalBlockPair(interval, create_block(input_text))
    new_block = pair.get_clean_block()
    assert new_block.lines[0].text == pair.block.lines[0].text


def test_clean_block_multiline(interval):
    """Test clean_block with multiple lines of text."""
    words1 = [TextWord(pymupdf.Rect(i, 0, i + 1, 1), w, page=1) for i, w in enumerate(["5-7.1", "m", "Clay", "and"])]
    words2 = [TextWord(pymupdf.Rect(0, 0, 0, 0), w, page=1) for w in ["silt", "with", "gravel"]]
    block = TextBlock([TextLine(words1), TextLine(words2)])

    pair = IntervalBlockPair(interval, block)
    new_block = pair.get_clean_block()

    assert new_block.lines[0].text == "Clay and"
    assert new_block.lines[0].rect == pymupdf.Rect(2, 0, 4, 1)  # first 2 words deleted
    assert new_block.lines[1].text == "silt with gravel"


def test_clean_block_empty_result(interval, create_block):
    """Test clean_block when cleaning would result in an empty line."""
    pair = IntervalBlockPair(interval, create_block("5-7.1m"))
    new_block = pair.get_clean_block()
    assert len(new_block.lines) == 0


@pytest.mark.parametrize(
    "test_interval,input_text",
    [
        (None, "5-7.1m Clay and silt"),
        (AToBInterval(None, DepthColumnEntry(7.1, pymupdf.Rect(1, 0, 2, 1))), "5-7.1m Clay and silt"),
    ],
)
def test_clean_block_missing_depth_info(create_block, test_interval, input_text):
    """Test clean_block with missing start or end depth information."""
    pair = IntervalBlockPair(test_interval, create_block(input_text))
    new_block = pair.get_clean_block()
    assert new_block.lines[0].text == pair.block.lines[0].text
