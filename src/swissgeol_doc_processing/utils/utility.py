"""Utility functions for the data extractor module."""

from pymupdf import Rect

from swissgeol_doc_processing.text.textline import TextLine


def get_lines_near_rect(
    search_left_factor: float,
    search_right_factor: float,
    search_above_factor: float,
    search_below_factor: float,
    lines: list[TextLine],
    rect: Rect,
) -> list[TextLine]:
    """Find the lines of the text that are close to a given rectangle.

    Args:
        search_left_factor (float): The factor to search to the left of the rectangle.
        search_right_factor (float): The factor to search to the right of the rectangle.
        search_above_factor (float): The factor to search above the rectangle.
        search_below_factor (float): The factor to search below the rectangle
        lines (list[TextLine]): Arbitrary text lines to search in.
        rect (pymupdf.Rect): The rectangle to search around.

    Returns:
        list[TextLine]: The lines close to the rectangle.
    """
    search_rect = Rect(
        rect.x0 - search_left_factor * rect.width,
        rect.y0 - search_above_factor * rect.height,
        rect.x1 + search_right_factor * rect.width,
        rect.y1 + search_below_factor * rect.height,
    )
    feature_lines = [line for line in lines if line.rect.intersects(search_rect)]

    return feature_lines
