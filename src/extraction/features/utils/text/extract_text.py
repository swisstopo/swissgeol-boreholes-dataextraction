"""Methods for extracting plain text from a PDF document."""

import pymupdf

from .textline import TextLine, TextWord


def extract_text_lines(page: pymupdf.Page) -> list[TextLine]:
    """Extract all text lines from the page.

    Sometimes, a single lines as identified by PyMuPDF, is still split into separate lines.

    Args:
        page (pymupdf.page): the page to extract text from

    Returns:
        list[TextLine]: A list of text lines.
    """
    return extract_text_lines_from_bbox(page, bbox=None)


def extract_text_lines_from_bbox(page: pymupdf.Page, bbox: pymupdf.Rect | None) -> list[TextLine]:
    """Extract all text lines from the page.

    Sometimes, a single lines as identified by PyMuPDF, is still split into separate lines.

    Args:
        page (pymupdf.page): the page to extract text from
        bbox (pymupdf.Rect | None): the bounding box to extract text from

    Returns:
        list[TextLine]: A list of text lines.
    """
    words = []
    words_by_line = {}
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in page.get_text("words", clip=bbox):
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        text_word = TextWord(rect, word, page.number + 1)
        words.append(text_word)
        key = f"{block_no}_{line_no}"
        if key not in words_by_line:
            words_by_line[key] = []
        words_by_line[key].append(text_word)

    raw_lines = [TextLine(words_by_line[key]) for key in words_by_line]

    lines = []
    current_line_words = []
    for line_index, raw_line in enumerate(raw_lines):
        for word_index, word in enumerate(raw_line.words):
            remaining_line = TextLine(raw_line.words[word_index:])
            if len(current_line_words) > 0 and remaining_line.is_line_start(lines, raw_lines[line_index + 1 :]):
                lines.append(TextLine(current_line_words))
                current_line_words = []
            current_line_words.append(word)
        if current_line_words:
            lines.append(TextLine(current_line_words))
            current_line_words = []

    return lines
