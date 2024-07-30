"""Test suite for the find_depth_columns module."""

import fitz
from stratigraphy.util.depthcolumn import BoundaryDepthColumn
from stratigraphy.util.depthcolumnentry import DepthColumnEntry


def test_boundarydepthcolumn_isarithmeticprogression():  # noqa: D103
    """Test the is_arithmetic_progression method of the BoundaryDepthColumn class."""
    page_number = 1
    column = BoundaryDepthColumn(
        [
            DepthColumnEntry(fitz.Rect(), value=1, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=2, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=3, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=4, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=5, page_number=page_number),
        ]
    )
    assert column.is_arithmetic_progression(), "The column should be recognized as arithmetic progression"

    column = BoundaryDepthColumn(
        [
            DepthColumnEntry(fitz.Rect(), value=17.6, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=18.15, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=18.65, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=19.3, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=19.9, page_number=page_number),
            DepthColumnEntry(fitz.Rect(), value=20.5, page_number=page_number),
        ]
    )
    assert not column.is_arithmetic_progression(), "The column should not be recognized as arithmetic progression"
