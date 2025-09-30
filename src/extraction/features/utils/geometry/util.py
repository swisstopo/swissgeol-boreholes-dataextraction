"""This module contains general utility functions for the stratigraphy module."""

from collections.abc import Callable

import pymupdf
from numpy.typing import ArrayLike

from .geometry_dataclasses import Circle, Line, Point


def axis_overlap(rect1: pymupdf.Rect, rect2: pymupdf.Rect, axis: str) -> float:
    """Calculate the overlap between two rectangles along a given axis ('x' or 'y').

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        axis (str): Axis along which to calculate overlap ('x' or 'y').

    Returns:
        float: The overlap between the two rectangles.
    """
    if axis == "x":
        a0, a1 = rect1.x0, rect1.x1
        b0, b1 = rect2.x0, rect2.x1
    elif axis == "y":
        a0, a1 = rect1.y0, rect1.y1
        b0, b1 = rect2.y0, rect2.y1
    else:
        raise ValueError("Axis must be 'x' or 'y'.")

    if a0 < b1 and b0 < a1:
        return min(a1, b1) - max(a0, b0)
    else:
        return 0.0


def axis_overlap_significant(
    rect1: pymupdf.Rect,
    rect2: pymupdf.Rect,
    axis: str,
    level: float,
    side_length_func: Callable[[float, float], float],
) -> bool:
    """Check if axis overlap is significant based on a comparison of rectangle sizes."""
    size1 = rect1.width if axis == "x" else rect1.height
    size2 = rect2.width if axis == "x" else rect2.height
    return axis_overlap(rect1, rect2, axis) > level * side_length_func(size1, size2)


# Now, small wrappers to be nice:
def x_overlap(rect1: pymupdf.Rect, rect2: pymupdf.Rect) -> float:
    """Calculate the x overlap between two rectangles.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.

    Returns:
        float: The x overlap between the two rectangles.
    """
    return axis_overlap(rect1, rect2, axis="x")


def y_overlap(rect1: pymupdf.Rect, rect2: pymupdf.Rect) -> float:
    """Calculate the y overlap between two rectangles.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.

    Returns:
        float: The y overlap between the two rectangles.
    """
    return axis_overlap(rect1, rect2, axis="y")


def x_overlap_significant_smallest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:
    """Check if the x overlap between two rectangles is significant relative to the width of the narrowest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return axis_overlap_significant(rect1, rect2, axis="x", level=level, side_length_func=min)


def x_overlap_significant_largest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:
    """Check if the x overlap between two rectangles is significant relative to the width of the widest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return axis_overlap_significant(rect1, rect2, axis="x", level=level, side_length_func=max)


def y_overlap_significant_smallest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:
    """Check if the y overlap between two rectangles is significant relative to the length of the narrowest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the y overlap is significant, otherwise False.
    """
    return axis_overlap_significant(rect1, rect2, axis="y", level=level, side_length_func=min)


def y_overlap_significant_largest(rect1: pymupdf.Rect, rect2: pymupdf.Rect, level: float) -> bool:
    """Check if the y overlap between two rectangles is significant relative to the length of the widest one.

    Args:
        rect1 (pymupdf.Rect): First rectangle.
        rect2 (pymupdf.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the y overlap is significant, otherwise False.
    """
    return axis_overlap_significant(rect1, rect2, axis="y", level=level, side_length_func=max)


def line_from_array(line: ArrayLike, scale_factor: float) -> Line:
    """Convert a line in the format of [[x1, y1, x2, y2]] to a Line objects.

    Args:
        line (ArrayLike): line as represented by an array of four numbers.
        scale_factor (float): The scale factor to apply to the lines. Required when
                              the pdf page was scaled before detecting lines.

    Returns:
        Line: The converted line.
    """
    start = Point(line[0][0] / scale_factor, line[0][1] / scale_factor)
    end = Point(line[0][2] / scale_factor, line[0][3] / scale_factor)

    return Line(start, end)


def circle_from_array(circle: ArrayLike, scale_factor: float) -> Circle:
    """Convert a circle in the format of [x, y, radius] to a Circle object.

    Args:
        circle (ArrayLike): circle as represented by an array of three numbers [x, y, radius].
        scale_factor (float): The scale factor to apply to the circles. Required when
                              the pdf page was scaled before detecting circles.

    Returns:
        Circle: The converted circle.
    """
    center = Point(circle[0] / scale_factor, circle[1] / scale_factor)
    radius = circle[2] / scale_factor

    return Circle(center, radius)


def compute_outer_rect(rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    """Compute the outer rectangle that contains all rectangles of each entry.

    Args:
        rects (list[pymupdf.Rect]): List of pymupdf.Rects.

    Returns:
        pymupdf.Rect: The minimal rectangle containing all input rectangles.
    """
    if not rects:
        raise ValueError("Entries list is empty.")

    # Start with the first rect
    outer_rect: pymupdf.Rect = rects[0]

    for rect in rects[1:]:
        outer_rect |= rect  # Expand the outer_rect to include entry.rect

    return outer_rect
