"""Test suite for the LayerDepths class."""

import pymupdf
from borehole_extraction.extraction.stratigraphy.layer.layer import DepthInterval
from borehole_extraction.extraction.stratigraphy.sidebar.sidebarentry import DepthColumnEntry


def test_line_anchor():  # noqa: D103
    """Test the line anchor property of the LayerDepths class."""
    start = DepthColumnEntry(value=5, rect=pymupdf.Rect(0, 0, 1, 1))
    end = DepthColumnEntry(value=10, rect=pymupdf.Rect(0, 2, 1, 3))
    assert DepthInterval(start, end).line_anchor == pymupdf.Point(1, 1.5), (
        "When the depth values are below each other, the line anchor should be halfway between the bottom-right of "
        "the start depth and the top-right of the end depth."
    )

    assert DepthInterval(start, end=None).line_anchor == pymupdf.Point(
        1, 1
    ), "When there is no end depth, the line anchor should be the bottom-right of the start depth."

    assert DepthInterval(start=None, end=end).line_anchor == pymupdf.Point(
        1, 2
    ), "When there is no start depth, the line anchor should be the top-right of the end depth."

    left = DepthColumnEntry(value=5, rect=pymupdf.Rect(0, 0, 1, 1))
    right = DepthColumnEntry(value=10, rect=pymupdf.Rect(2, 0, 3, 1))
    assert DepthInterval(left, right).line_anchor == pymupdf.Point(3, 0.5), (
        "When the depth entries are next to each other, the line anchor should on the right-hand-side of the "
        "right-most rect."
    )


def test_background_rect():  # noqa: D103
    """Test the background_rect property of the LayerDepths class."""
    start = DepthColumnEntry(value=5, rect=pymupdf.Rect(0, 0, 1, 1))
    end = DepthColumnEntry(value=10, rect=pymupdf.Rect(0, 2, 1, 3))
    depths = DepthInterval(start, end)
    assert depths.background_rect == pymupdf.Rect(
        start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0
    ), "The background rect should be (0, 1, 1, 2)"
    assert depths.background_rect == pymupdf.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"

    rect = pymupdf.Rect(0, 0, 1, 1)
    start = DepthColumnEntry(value=5, rect=rect)
    end = DepthColumnEntry(value=10, rect=rect)
    assert (
        DepthInterval(start, end).background_rect is None
    ), "When start and end depths are overlapping, there should be no background rect."
