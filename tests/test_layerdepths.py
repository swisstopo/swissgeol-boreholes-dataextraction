"""Test suite for the LayerDepths class."""

import pymupdf

from extraction.features.stratigraphy.layer.layer import LayerDepths, LayerDepthsEntry


def test_line_anchor():  # noqa: D103
    """Test the line anchor property of the LayerDepths class."""
    start = LayerDepthsEntry(5, pymupdf.Rect(0, 0, 1, 1), 0)
    end = LayerDepthsEntry(10, pymupdf.Rect(0, 2, 1, 3), 0)
    assert LayerDepths(start, end).get_line_anchor(0) == pymupdf.Point(1, 1.5), (
        "When the depth values are below each other, the line anchor should be halfway between the bottom-right of "
        "the start depth and the top-right of the end depth."
    )

    assert LayerDepths(start, end=None).get_line_anchor(0) == pymupdf.Point(1, 1), (
        "When there is no end depth, the line anchor should be the bottom-right of the start depth."
    )

    assert LayerDepths(start=None, end=end).get_line_anchor(0) == pymupdf.Point(1, 2), (
        "When there is no start depth, the line anchor should be the top-right of the end depth."
    )

    left = LayerDepthsEntry(5, pymupdf.Rect(0, 0, 1, 1), 0)
    right = LayerDepthsEntry(10, pymupdf.Rect(2, 0, 3, 1), 0)
    assert LayerDepths(left, right).get_line_anchor(0) == pymupdf.Point(3, 0.5), (
        "When the depth entries are next to each other, the line anchor should on the right-hand-side of the "
        "right-most rect."
    )
    end_next_page = LayerDepthsEntry(10, pymupdf.Rect(0, 0, 1, 1), 1)
    assert LayerDepths(start, end_next_page).get_line_anchor(0) == pymupdf.Point(1, 1), (
        "When depths are across 2 pages, the anchor queried at the same page as the start is should point at the start"
    )
    assert LayerDepths(start, end_next_page).get_line_anchor(1) == pymupdf.Point(1, 0), (
        "When depths are across 2 pages, the anchor queried at the same page as the end should be at the top "
        "of the second page to signal that the description spans 2 pages"
    )


def test_background_rect():  # noqa: D103
    """Test the background_rect property of the LayerDepths class."""
    start = LayerDepthsEntry(5, pymupdf.Rect(0, 0, 1, 1), 0)
    end = LayerDepthsEntry(10, pymupdf.Rect(0, 2, 1, 3), 0)
    depths = LayerDepths(start, end)
    assert depths.get_background_rect(0) == pymupdf.Rect(start.rect.x0, start.rect.y1, start.rect.x1, end.rect.y0), (
        "The background rect should be (0, 1, 1, 2)"
    )
    assert depths.get_background_rect(0) == pymupdf.Rect(0, 1, 1, 2), "The background rect should be (0, 1, 1, 2)"

    rect = pymupdf.Rect(0, 0, 1, 1)
    start = LayerDepthsEntry(5, rect, 0)
    end = LayerDepthsEntry(10, rect, 0)
    assert LayerDepths(start, end).get_background_rect(0) is None, (
        "When start and end depths are overlapping, there should be no background rect."
    )

    # special case when depths are spanning 2 pages
    start = LayerDepthsEntry(5, pymupdf.Rect(0, 1, 1, 2), 0)
    end = LayerDepthsEntry(10, pymupdf.Rect(0, 0.5, 1, 1), 1)
    depths = LayerDepths(start, end)
    assert depths.get_background_rect(0) == pymupdf.Rect(0, 2, 1, 10000), "From the start to the bottom of the page"
    assert depths.get_background_rect(1) == pymupdf.Rect(0, 0, 1, 0.5), "From the top of the page to the end"
