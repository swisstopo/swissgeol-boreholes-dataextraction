"""Methods for extracting plain text from a PDF document."""

import fitz

from stratigraphy.util.line import TextLine, TextWord


def extract_text_lines(page: fitz.Page) -> list[TextLine]:
    """Extract all text lines from the page.

    Sometimes, a single lines as identified by PyMuPDF, is still split into separate lines.

    Args:
        page (fitz.page): the page to extract text from
        page_number (int): the page number (first page is 1)

    Returns:
        list[TextLine]: A list of text lines.
    """
    words = []
    words_by_line = {}
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in fitz.utils.get_text(page, "words"):
        rect = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
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
