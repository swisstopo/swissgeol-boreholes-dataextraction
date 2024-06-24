"""Test suite for the find_depth_columns module."""

import fitz
from stratigraphy.util.depthcolumn import BoundaryDepthColumn
from stratigraphy.util.depthcolumnentry import DepthColumnEntry


def test_boundarydepthcolumn_isarithmeticprogression():  # noqa: D103
    column = BoundaryDepthColumn(
        [
            DepthColumnEntry(fitz.Rect(), value=1),
            DepthColumnEntry(fitz.Rect(), value=2),
            DepthColumnEntry(fitz.Rect(), value=3),
            DepthColumnEntry(fitz.Rect(), value=4),
            DepthColumnEntry(fitz.Rect(), value=5),
        ]
    )
    assert column.is_arithmetic_progression(), "The column should be recognized as arithmetic progression"

    column = BoundaryDepthColumn(
        [
            DepthColumnEntry(fitz.Rect(), value=17.6),
            DepthColumnEntry(fitz.Rect(), value=18.15),
            DepthColumnEntry(fitz.Rect(), value=18.65),
            DepthColumnEntry(fitz.Rect(), value=19.3),
            DepthColumnEntry(fitz.Rect(), value=19.9),
            DepthColumnEntry(fitz.Rect(), value=20.5),
        ]
    )
    assert not column.is_arithmetic_progression(), "The column should not be recognized as arithmetic progression"
