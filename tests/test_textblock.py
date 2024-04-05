"""Test suite for the textblock module."""

import fitz
from stratigraphy.util.line import DepthInterval, TextLine
from stratigraphy.util.textblock import TextBlock


def test_concatenate():  # noqa: D103
    tb1 = TextBlock([TextLine([DepthInterval(fitz.Rect([0, 0, 5, 1]), "Hello")])])
    tb2 = TextBlock([TextLine([DepthInterval(fitz.Rect([0, 1, 5, 2]), "World")])])
    tb3 = tb1.concatenate(tb2)
    assert len(tb3.lines) == 2, "There should be 2 lines in the concatenated TextBlock"
    assert tb3.text == "Hello World", "The text should be 'Hello World'"


def test_split_based_on_indentation():  # noqa: D103
    tb = TextBlock(
        [
            TextLine([DepthInterval(fitz.Rect(0, 0, 20, 5), "Hello")]),
            TextLine([DepthInterval(fitz.Rect(0, 8, 20, 13), "Hello")]),
            TextLine([DepthInterval(fitz.Rect(3, 16, 22, 21), "World")]),  # Indented line
        ]
    )
    blocks = tb.split_based_on_indentation()
    assert len(blocks) == 2, "There should be 2 blocks after splitting"


def test_post_init():  # noqa: D103
    tb = TextBlock([TextLine([DepthInterval(fitz.Rect(0, 0, 5, 1), "Hello")])])
    assert tb.line_count == 1, "The line count should be 1"
    assert tb.text == "Hello", "The text should be 'Hello'"
    assert tb.rect == fitz.Rect(0, 0, 5, 1), "The rect should be the same as the line's rect"
