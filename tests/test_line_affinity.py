"""Test suite for the line affinity calculator module."""

import pymupdf
import pytest

from extraction.features.extract import get_pairs_based_on_line_affinity
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from swissgeol_doc_processing.text.textline_affinity import get_line_affinity

page_number = 1
textline1 = TextLine([TextWord(pymupdf.Rect([0, 0, 10, 10]), "Hello", page_number)])
textline2 = TextLine([TextWord(pymupdf.Rect([0, 9, 10, 19]), "World", page_number)])
textline3 = TextLine(
    [TextWord(pymupdf.Rect([0, 37, 10, 47]), "Hey", page_number)]
)  # larger vertical distance to previous blocks
description_lines = [textline1, textline2, textline3]

geometric_lines_cut = [Line(Point(-5, 11), Point(10, 11))]  # spanning line cuts the first and second line
geometric_lines_lefthandside = [Line(Point(-1, 11), Point(3, 11))]  # left-hand side line cuts the same lines

material_description_rect = pymupdf.Rect(0, 0, 5, 42)
block_line_ratio = 0.5
left_line_length_threshold = 3


@pytest.mark.parametrize(
    "geometrical_lines,expected_num_block",
    [
        pytest.param([], 2, id="no_geom_lines"),
        pytest.param(geometric_lines_cut, 3, id="span_line"),
        pytest.param(geometric_lines_lefthandside, 3, id="left_line"),
    ],
)
def test_get_description_blocks(geometrical_lines, expected_num_block):
    """Test the grouping of description lines into blocks.

    Tests three scenarios:
        1. no geometrical lines: expect 2 blocks (first two lines together as they slightly overlap, third separate)
        2. span line cutting through first two lines: expect 3 blocks (all separate)
        3. left-hand side line cutting through first two lines: expect 3 blocks (all separate)
    """
    line_affinities = get_line_affinity(
        description_lines,
        material_description_rect,
        geometrical_lines,
        block_line_ratio,
        left_line_length_threshold,
    )
    pairs = get_pairs_based_on_line_affinity(description_lines, line_affinities)

    assert len(pairs) == expected_num_block
