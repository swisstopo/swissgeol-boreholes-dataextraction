"""Test suite for the find_description module."""

import fitz
from stratigraphy.lines.line import TextLine, TextWord
from stratigraphy.text.find_description import get_description_blocks
from stratigraphy.util.dataclasses import Line, Point

page_number = 1
textline1 = TextLine([TextWord(fitz.Rect([0, 0, 10, 10]), "Hello", page_number)])
textline2 = TextLine([TextWord(fitz.Rect([0, 15, 10, 25]), "World", page_number)])
textline3 = TextLine(
    [TextWord(fitz.Rect([0, 37, 10, 47]), "Hey", page_number)]
)  # larger vertical distance to previous blocks

geometric_lines = [Line(Point(500, 1), Point(505, 1))]  # line does not cut the blocks
geometric_lines_cut = [Line(Point(-5, 12), Point(10, 12))]  # line cuts the first and second line
geometric_lines_lefthandside = [Line(Point(-1, 30), Point(3, 30))]  # line cuts the second and third line

description_lines = [
    textline1,
    textline2,
    textline3,
]
material_description_rect = fitz.Rect(0, 0, 5, 42)
block_line_ratio = 0.5
left_line_length_threshold = 3


def test_get_description_blocks():  # noqa: D103
    """Test the grouping of description lines into blocks."""
    target_layer_count = 2  # expect two blocks. But the line do not cut the blocks
    blocks = get_description_blocks(
        description_lines,
        geometric_lines,
        material_description_rect,
        block_line_ratio,
        left_line_length_threshold,
        target_layer_count,
    )

    assert len(blocks) == 2, "There should be 3 blocks"
    assert blocks[0].text == "Hello World", "The first block should contain the text 'Hello'"
    assert blocks[1].text == "Hey", "The second block should contain the text 'World'"


def test_get_description_blocks_separated_by_line():  # noqa: D103
    """Test the splitting of blocks based on the presence of a line."""
    target_layer_count = 1  # should not trigger splitting the blocks with vertical distances
    blocks = get_description_blocks(
        description_lines,
        geometric_lines_cut,
        material_description_rect,
        block_line_ratio,
        left_line_length_threshold,
        target_layer_count,
    )

    assert len(blocks) == 2, "There should be 2 blocks"
    assert blocks[0].text == "Hello", "The first block should contain the text 'Hello'"
    assert blocks[1].text == "World Hey", "The second block should contain the text 'World Hey'"


def test_get_description_blocks_separated_by_lefthandside_line():  # noqa: D103
    """Test the splitting of blocks based on the presence of a lefthandside line."""
    target_layer_count = 1  # only one block, but the lefthand line still cuts them into two blocks
    geometric_lines_all = geometric_lines_cut + geometric_lines_lefthandside
    blocks = get_description_blocks(
        description_lines,
        geometric_lines_all,
        material_description_rect,
        block_line_ratio,
        left_line_length_threshold,
        target_layer_count,
    )

    assert len(blocks) == 3, "There should be 3 blocks"
    assert blocks[0].text == "Hello", "The first block should contain the text 'Hello'"
    assert blocks[1].text == "World", "The second block should contain the text 'World'"
    assert blocks[2].text == "Hey", "The second block should contain the text 'Hey'"
