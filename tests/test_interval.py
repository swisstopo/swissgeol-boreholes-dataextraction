"""Test suite for the interval module."""

import fitz
from stratigraphy.depthcolumnentry.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.interval import AAboveBInterval, AToBInterval


def test_line_anchor():  # noqa: D103
    """Test the line anchor property of the AAboveBInterval and AToBInterval classes."""
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    a_above_b_interval = AAboveBInterval(start, end)
    assert a_above_b_interval.line_anchor == fitz.Point(1, 1.5), (
        "The 'line anchor' for an AAboveBInterval should be halfway between the bottom-right of the start depth and "
        "the top-right of the end depth."
    )

    a_above_b_interval = AAboveBInterval(start, end=None)
    assert a_above_b_interval.line_anchor == fitz.Point(
        1, 1
    ), "The 'line anchor' for an AAboveBInterval without end should be the bottom-right of the start depth."

    a_above_b_interval = AAboveBInterval(start=None, end=end)
    assert a_above_b_interval.line_anchor == fitz.Point(
        1, 2
    ), "The 'line anchor' for a AAboveBInterval without start should be the top-right of the end depth."

    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(2, 0, 3, 1), 10)
    a_to_b_interval = AToBInterval(start, end)
    assert a_to_b_interval.line_anchor == fitz.Point(
        3, 0.5
    ), "The 'line anchor' for an AToBInterval should be the midpoint of the right-hand-side of the end rect."


def test_background_rect():  # noqa: D103
    """Test the background_rect property of the AAboveBInterval class."""
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    a_above_b_interval = AAboveBInterval(start, end)
    assert a_above_b_interval.background_rect == fitz.Rect(
        start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0
    ), "The background rect should be (0, 1, 1, 2)"
    assert a_above_b_interval.background_rect == fitz.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"


# TODO: add tests for AAboveBInterval.matching_blocks
