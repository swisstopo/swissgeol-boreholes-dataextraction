"""Test suite for the DepthColumnEntry class."""

import pymupdf
from borehole_extraction.extraction.stratigraphy.base_sidebar_entry.sidebar_entry import DepthColumnEntry


def test_from_string_value():
    """Test the DepthColumnEntry.from_string_value function."""
    rect = pymupdf.Rect(0, 0, 10, 10)

    def run_test(string_in, value_out, has_decimal_point):
        depth_entry = DepthColumnEntry.from_string_value(rect, string_in)
        assert depth_entry.value == value_out
        assert depth_entry.has_decimal_point is has_decimal_point

    run_test("12", 12.0, False)
    run_test("12.5", 12.5, True)
    run_test("12.", 12.0, True)
    run_test(".5", 0.5, True)
    run_test("-7", 7.0, False)
