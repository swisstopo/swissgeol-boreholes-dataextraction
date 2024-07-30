"""Test suite for the textblock module."""

import fitz
from stratigraphy.util.line import TextLine, TextWord
from stratigraphy.util.textblock import TextBlock, block_distance


def test_concatenate():  # noqa: D103
    """Test the concatenation of two TextBlocks."""
    page_number = 1
    tb1 = TextBlock([TextLine([TextWord(fitz.Rect([0, 0, 5, 1]), "Hello", page_number)])])
    tb2 = TextBlock([TextLine([TextWord(fitz.Rect([0, 1, 5, 2]), "World", page_number)])])
    tb3 = tb1.concatenate(tb2)
    assert len(tb3.lines) == 2, "There should be 2 lines in the concatenated TextBlock"
    assert tb3.text == "Hello World", "The text should be 'Hello World'"


def test_split_based_on_indentation():  # noqa: D103
    """Test the splitting of a TextBlock based on indentation."""
    page_number = 1
    tb = TextBlock(
        [
            TextLine([TextWord(fitz.Rect(0, 0, 20, 5), "Hello", page_number)]),
            TextLine([TextWord(fitz.Rect(0, 8, 20, 13), "Hello", page_number)]),
            TextLine([TextWord(fitz.Rect(3, 16, 22, 21), "World", page_number)]),  # Indented line
        ]
    )
    blocks = tb.split_based_on_indentation()
    assert len(blocks) == 2, "There should be 2 blocks after splitting"


def test_post_init():  # noqa: D103
    """Test the post-init method of the TextBlock class."""
    page_number = 1
    tb = TextBlock([TextLine([TextWord(fitz.Rect(0, 0, 5, 1), "Hello", page_number)])])
    assert tb.line_count == 1, "The line count should be 1"
    assert tb.text == "Hello", "The text should be 'Hello'"
    assert tb.rect == fitz.Rect(0, 0, 5, 1), "The rect should be the same as the line's rect"


def test_post_init_longer_text():  # noqa: D103
    """Test the post-init method of the TextBlock class with multiple lines."""
    page_number = 1
    tb = TextBlock(
        [
            TextLine([TextWord(fitz.Rect(0, 0, 5, 1), "Hello", page_number)]),
            TextLine(
                [
                    TextWord(fitz.Rect(0, 1, 5, 2), "It's", page_number),
                    TextWord(fitz.Rect(5, 1, 10, 2), "me", page_number),
                ]
            ),
        ]
    )
    assert tb.line_count == 2, "The line count should be 2"
    assert tb.text == "Hello It's me", "Lines and words should be joined with a space"
    assert tb.rect == fitz.Rect(0, 0, 10, 2), "The block's rect should surround the rects of the individual words"


def test_block_distance():  # noqa: D103
    """Test the calculation of the distance between two TextBlocks."""
    page_number = 1
    block_1 = TextBlock([TextLine([TextWord(fitz.Rect(0, 0, 5, 1), "Hello", page_number)])])
    block_2 = TextBlock([TextLine([TextWord(fitz.Rect(0, 2, 5, 3), "Hello", page_number)])])
    assert (
        block_distance(block_1, block_2) == 1
    ), "The distance should be measured from the bottom of the first block to the top of the second block."
    assert (
        block_distance(block_2, block_1) == -3
    ), "The distance should negative when the second block is above the first block."
