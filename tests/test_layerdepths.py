"""Test suite for the LayerDepths class."""

import fitz
from stratigraphy.layer.layer import LayerDepths, LayerDepthsEntry


def test_line_anchor():  # noqa: D103
    """Test the line anchor property of the AAboveBInterval and AToBInterval classes."""
    start = LayerDepthsEntry(5, fitz.Rect(0, 0, 1, 1))
    end = LayerDepthsEntry(10, fitz.Rect(0, 2, 1, 3))
    depths = LayerDepths(start, end)
    assert depths.line_anchor == fitz.Point(1, 1.5), (
        "The 'line anchor' for an AAboveBInterval should be halfway between the bottom-right of the start depth and "
        "the top-right of the end depth."
    )

    depths = LayerDepths(start, end=None)
    assert depths.line_anchor == fitz.Point(
        1, 1
    ), "The 'line anchor' for an AAboveBInterval without end should be the bottom-right of the start depth."

    depths = LayerDepths(start=None, end=end)
    assert depths.line_anchor == fitz.Point(
        1, 2
    ), "The 'line anchor' for a AAboveBInterval without start should be the top-right of the end depth."

    start = LayerDepthsEntry(5, fitz.Rect(0, 0, 1, 1))
    end = LayerDepthsEntry(10, fitz.Rect(2, 0, 3, 1))
    depths = LayerDepths(start, end)
    assert depths.line_anchor == fitz.Point(
        3, 0.5
    ), "The 'line anchor' for an AToBInterval should be the midpoint of the right-hand-side of the end rect."


def test_background_rect():  # noqa: D103
    """Test the background_rect property of the AAboveBInterval class."""
    start = LayerDepthsEntry(5, fitz.Rect(0, 0, 1, 1))
    end = LayerDepthsEntry(10, fitz.Rect(0, 2, 1, 3))
    depths = LayerDepths(start, end)
    assert depths.background_rect == fitz.Rect(
        start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0
    ), "The background rect should be (0, 1, 1, 2)"
    assert depths.background_rect == fitz.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"
