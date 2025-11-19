"""This module contains functions to find the description of a material in a pdf page."""

import pymupdf

from swissgeol_doc_processing.text.textline import TextLine


def get_description_lines(lines: list[TextLine], material_description_rect: pymupdf.Rect) -> list[TextLine]:
    """Get the description lines of a material.

    Checks if the lines are within the material description rectangle and if they are not too far to the right.

    Args:
        lines (list[TextLine]): The lines to filter.
        material_description_rect (pymupdf.Rect): The rectangle containing the material description.

    Returns:
        list[TextLine]: The filtered lines.
    """
    if not lines:
        return []
    filtered_lines = [
        line
        for line in lines
        if line.rect.x0 < material_description_rect.x1 - 0.4 * material_description_rect.width
        if material_description_rect.contains(line.rect)
    ]
    return sorted([line for line in filtered_lines if line], key=lambda line: line.rect.y0)
