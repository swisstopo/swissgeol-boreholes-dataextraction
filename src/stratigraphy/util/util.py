"""This module contains general utility functions for the stratigraphy module."""

import re
from collections.abc import MutableMapping

import fitz
import yaml
from numpy.typing import ArrayLike

from stratigraphy import PROJECT_ROOT
from stratigraphy.util.dataclasses import Line, Point


def x_overlap(rect1: fitz.Rect, rect2: fitz.Rect) -> float:  # noqa: D103
    """Calculate the x overlap between two rectangles.

    Args:
        rect1 (fitz.Rect): First rectangle.
        rect2 (fitz.Rect): Second rectangle.

    Returns:
        float: The x overlap between the two rectangles.
    """
    if (rect1.x0 < rect2.x1) and (rect2.x0 < rect1.x1):
        return min(rect1.x1, rect2.x1) - max(rect1.x0, rect2.x0)
    else:
        return 0


def x_overlap_significant_smallest(rect1: fitz.Rect, rect2: fitz.Rect, level: float) -> bool:  # noqa: D103
    """Check if the x overlap between two rectangles is significant relative to the width of the narrowest one.

    Args:
        rect1 (fitz.Rect): First rectangle.
        rect2 (fitz.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return x_overlap(rect1, rect2) > level * min(rect1.width, rect2.width)


def x_overlap_significant_largest(rect1: fitz.Rect, rect2: fitz.Rect, level: float) -> bool:  # noqa: D103
    """Check if the x overlap between two rectangles is significant relative to the width of the widest one.

    Args:
        rect1 (fitz.Rect): First rectangle.
        rect2 (fitz.Rect): Second rectangle.
        level (float): Level of significance.

    Returns:
        bool: True if the x overlap is significant, otherwise False.
    """
    return x_overlap(rect1, rect2) > level * max(rect1.width, rect2.width)


def flatten(dictionary: dict, parent_key: str = "", separator: str = "__") -> dict:
    """Flatten a nested dictionary.

    Args:
        dictionary (Dict): Dictionary to flatten.
        parent_key (str, optional): Prefix for flattened key. Defaults to ''.
        separator (str, optional): The separator used when concatenating keys. Defaults to '__'.

    Returns:
        Dict: Flattened dictionary.
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


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


def read_params(params_name: str) -> dict:
    """Read parameters from a yaml file.

    Args:
        params_name (str): Name of the params yaml file.
    """
    with open(PROJECT_ROOT / "config" / params_name) as f:
        params = yaml.safe_load(f)

    return params


def parse_text(text: str) -> str:
    """Parse text by removing non-alphanumeric characters and converting to lowercase.

    Args:
        text (str): Text to parse.

    Returns:
        str: Parsed text.
    """
    not_alphanum = re.compile(r"[^\w\d]", re.U)
    return not_alphanum.sub("", text).lower()
