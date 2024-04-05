"""Test suite for the interval module."""

import fitz
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.interval import BoundaryInterval


def test_line_anchor():  # noqa: D103
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    boundary_interval = BoundaryInterval(start, end)
    assert boundary_interval.line_anchor == fitz.Point(1, 1.5), "The line anchor should be at (1, 1.5)"


def test_background_rect():  # noqa: D103
    start = DepthColumnEntry(fitz.Rect(0, 0, 1, 1), 5)
    end = DepthColumnEntry(fitz.Rect(0, 2, 1, 3), 10)
    boundary_interval = BoundaryInterval(start, end)
    assert boundary_interval.background_rect == fitz.Rect(
        start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0
    ), "The background rect should be (0, 1, 1, 2)"
    assert boundary_interval.background_rect == fitz.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"


# TODO: add tests for BoundaryInterval.matching_blocks
