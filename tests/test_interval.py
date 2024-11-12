"""Test suite for the interval module."""

import fitz
from stratigraphy.depthcolumn.depthcolumnentry import AToBDepthColumnEntry, DepthColumnEntry
from stratigraphy.util.interval import AAboveBInterval, AToBInterval


def test_line_anchor():  # noqa: D103
    """Test the line anchor property of the BoundaryInterval and LayerInterval classes."""
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    boundary_interval = AAboveBInterval(start, end)
    assert boundary_interval.line_anchor == fitz.Point(1, 1.5), (
        "The 'line anchor' for a BoundaryInterval should be halfway between the bottom-right of the start depth and "
        "the top-right of the end depth."
    )

    boundary_interval = AAboveBInterval(start, end=None)
    assert boundary_interval.line_anchor == fitz.Point(
        1, 1
    ), "The 'line anchor' for a BoundaryInterval without end should be the bottom-right of the start depth."

    boundary_interval = AAboveBInterval(start=None, end=end)
    assert boundary_interval.line_anchor == fitz.Point(
        1, 2
    ), "The 'line anchor' for a BoundaryInterval without start should be the top-right of the end depth."

    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(2, 0, 3, 1), 10)
    entry = AToBDepthColumnEntry(start, end)
    layer_interval = AToBInterval(entry)
    assert layer_interval.line_anchor == fitz.Point(
        3, 0.5
    ), "The 'line anchor' for a LayerInterval should be the midpoint of the right-hand-side of the end rect."


def test_background_rect():  # noqa: D103
    """Test the background_rect property of the BoundaryInterval class."""
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    boundary_interval = AAboveBInterval(start, end)
    assert boundary_interval.background_rect == fitz.Rect(
        start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0
    ), "The background rect should be (0, 1, 1, 2)"
    assert boundary_interval.background_rect == fitz.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"


# TODO: add tests for BoundaryInterval.matching_blocks
