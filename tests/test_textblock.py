"""Test suite for the textblock module."""

import pymupdf

from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine, TextWord


def test_concatenate():  # noqa: D103
    """Test the concatenation of two TextBlocks."""
    page_number = 1
    tb1 = TextBlock([TextLine([TextWord(pymupdf.Rect([0, 0, 5, 1]), "Hello", page_number)])])
    tb2 = TextBlock([TextLine([TextWord(pymupdf.Rect([0, 1, 5, 2]), "World", page_number)])])
    tb3 = tb1.concatenate(tb2)
    assert len(tb3.lines) == 2, "There should be 2 lines in the concatenated TextBlock"
    assert tb3.text == "Hello World", "The text should be 'Hello World'"


def test_post_init():  # noqa: D103
    """Test the post-init method of the TextBlock class."""
    page_number = 1
    tb = TextBlock([TextLine([TextWord(pymupdf.Rect(0, 0, 5, 1), "Hello", page_number)])])
    assert tb.line_count == 1, "The line count should be 1"
    assert tb.text == "Hello", "The text should be 'Hello'"
    assert tb.rect == pymupdf.Rect(0, 0, 5, 1), "The rect should be the same as the line's rect"


def test_post_init_longer_text():  # noqa: D103
    """Test the post-init method of the TextBlock class with multiple lines."""
    page_number = 1
    tb = TextBlock(
        [
            TextLine([TextWord(pymupdf.Rect(0, 0, 5, 1), "Hello", page_number)]),
            TextLine(
                [
                    TextWord(pymupdf.Rect(0, 1, 5, 2), "It's", page_number),
                    TextWord(pymupdf.Rect(5, 1, 10, 2), "me", page_number),
                ]
            ),
        ]
    )
    assert tb.line_count == 2, "The line count should be 2"
    assert tb.text == "Hello It's me", "Lines and words should be joined with a space"
    assert tb.rect == pymupdf.Rect(0, 0, 10, 2), "The block's rect should surround the rects of the individual words"
