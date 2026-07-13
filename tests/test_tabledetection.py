"""Test module for detection of table-like structures."""

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.table_detection import detect_table_structures

table_detection_params = read_params("table_detection_params.yml")


def test_table_detection_lines() -> None:
    """Test simple cases for table detection."""
    table_detection_params_modified = table_detection_params.copy()
    table_detection_params_modified["tables"]["min_confidence"] = 0

    x0 = 10
    x1 = 190
    y0 = 10
    y1 = 190
    top_left = Point(x0, y0)
    top_right = Point(x1, y0)
    bottom_left = Point(x0, y1)
    bottom_right = Point(x1, y1)

    top = Line(top_left, top_right)
    bottom = Line(bottom_left, bottom_right)
    left = Line(top_left, bottom_left)
    right = Line(top_right, bottom_right)
    diag1 = Line(top_left, bottom_right)
    diag2 = Line(bottom_left, top_right)
    short = Line(Point(x0, y0 + 10), Point(x0 + 10, y0 + 10))
    lines = [top, bottom, left, right, diag1, diag2, short]

    tables = detect_table_structures(
        page_width=200,
        page_height=200,
        geometric_lines=lines,
        text_lines=[],
        table_detection_params=table_detection_params_modified,
    )
    assert len(tables) == 1, "There should be one detected table"
    table = tables[0]

    assert set(table.vertical_lines) == {left, right}, "The two vertical lines should be the left and right lines"
    assert set(table.horizontal_lines) == {top, bottom}, "The two horizontal lines should be the top and bottom lines"

    lines_without_bottom = lines.copy()
    lines_without_bottom.remove(bottom)
    tables = detect_table_structures(
        page_width=200,
        page_height=200,
        geometric_lines=lines_without_bottom,
        text_lines=[],
        table_detection_params=table_detection_params_modified,
    )
    assert len(tables) == 0, "There should no detected table after excluding the bottom line"

    lines_without_right = lines.copy()
    lines_without_right.remove(right)
    tables = detect_table_structures(
        page_width=200,
        page_height=200,
        geometric_lines=lines_without_right,
        text_lines=[],
        table_detection_params=table_detection_params_modified,
    )
    assert len(tables) == 0, "There should no detected table after excluding the right line"
