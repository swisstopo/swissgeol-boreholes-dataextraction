"""Detects groundwater readings highlighted by a distinct text color.

Some documents mark the groundwater reading in a color that differs from the rest of the page
(e.g. blue against otherwise-black text).
"""

from collections import Counter

from extraction.features.groundwater.utility import extract_date
from swissgeol_doc_processing.text.textline import TextLine


def get_minority_color_lines(text_lines: list[TextLine]) -> list[TextLine]:
    """Find text lines whose color differs from the page's dominant color and that contain a date.

    The page's most common word color is treated as "normal" (usually, but not always, black).
    Any line rendered in a different color is only considered a groundwater candidate if it also
    contains a date, to avoid flagging unrelated colored annotations (e.g. other highlighted
    elevations, watermarks, or documents that render all their text in a uniform non-black color).

    Args:
        text_lines (list[TextLine]): The text lines already extracted for this page.

    Returns:
        list[TextLine]: Lines in a minority color that also contain a date.
    """
    word_colors = [word.color for line in text_lines for word in line.words]
    if not word_colors:
        return []

    modal_color = Counter(word_colors).most_common(1)[0][0]

    def line_color(text_line: TextLine) -> int | None:
        """Determine the color of a text line from the color of its words."""
        return next((word.color for word in text_line.words), modal_color)

    return [line for line in text_lines if line_color(line) != modal_color and extract_date(line.text)[0] is not None]
