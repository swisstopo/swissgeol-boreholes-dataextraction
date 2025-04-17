"""This module contains general utility functions for the stratigraphy module."""

import pymupdf
from borehole_extraction.extraction.util_extraction.dataclasses import Line, Point
from numpy.typing import ArrayLike


def x_overlap(rect1: pymupdf.Rect, rect2: pymupdf.Rect) -> float:  # noqa: D103
    """Calculate the x overlap between two rectangles.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.

    Returns:
        float: The x overlap between the two rectangles.
    """
    if (rect1.x0 < rect2.x1) and (rect2.x0 < rect1.x1):
        return min(rect1.x1, rect2.x1) - max(rect1.x0, rect2.x0)
    else:
        return 0


def x_overlap_significant_smallest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:  # noqa: D103
    """Check if the x overlap between two rectangles is significant relative to the width of the narrowest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return x_overlap(rect1, rect2) > level * min(rect1.width, rect2.width)


def x_overlap_significant_largest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:  # noqa: D103
    """Check if the x overlap between two rectangles is significant relative to the width of the widest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return x_overlap(rect1, rect2) > level * max(rect1.width, rect2.width)


def line_from_array(line: ArrayLike, scale_factor: float) -> Line:
    """Convert a line in the format of [[x1, y1, x2, y2]] to a Line objects.

    Args:
        line (ArrayLike): line as represented by an array of four numbers.
        scale_factor (float): The scale factor to apply to the lines. Required when
                              the pdf page was scaled before detecting lines.

    Returns:
        Line: The converted line.
    """
    start = Point(int(line[0][0] / scale_factor), int(line[0][1] / scale_factor))
    end = Point(int(line[0][2] / scale_factor), int(line[0][3] / scale_factor))
    return Line(start, end)
