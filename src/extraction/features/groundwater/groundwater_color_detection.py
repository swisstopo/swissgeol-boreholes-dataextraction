"""Detects groundwater readings highlighted by a distinct text color.

Some documents mark the groundwater reading in a color that differs from the rest of the page
(e.g. blue against otherwise-black text).
"""

from collections import Counter

import pymupdf

from extraction.features.groundwater.utility import extract_date
from swissgeol_doc_processing.text.textline import TextLine


def get_minority_color_lines(page: pymupdf.Page, text_lines: list[TextLine]) -> list[TextLine]:
    """Find text lines whose color differs from the page's dominant color and that contain a date.

    The page's most common span color is treated as "normal" (usually, but not always, black).
    Any line rendered in a different color is only considered a groundwater candidate if it also
    contains a date, to avoid flagging unrelated colored annotations (e.g. other highlighted
    elevations, watermarks, or documents that render all their text in a uniform non-black color).

    Args:
        page (pymupdf.Page): The page to read span colors from.
        text_lines (list[TextLine]): The text lines already extracted for this page.

    Returns:
        list[TextLine]: Lines in a minority color that also contain a date.
    """
    span_colors: list[tuple[pymupdf.Rect, int]] = [
        (pymupdf.Rect(span["bbox"]), span.get("color"))
        for block in page.get_text("rawdict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    if not span_colors:
        return []

    modal_color = Counter(color for _, color in span_colors).most_common(1)[0][0]

    def line_color(text_line: TextLine) -> int | None:
        return next((color for rect, color in span_colors if rect.intersects(text_line.rect)), modal_color)

    return [line for line in text_lines if line_color(line) != modal_color and extract_date(line.text)[0] is not None]
