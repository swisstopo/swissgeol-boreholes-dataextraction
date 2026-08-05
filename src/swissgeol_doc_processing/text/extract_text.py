"""Methods for extracting plain text from a PDF document."""

import math

import pymupdf

from swissgeol_doc_processing.text.textline import TextLine, TextWord


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
    raw_lines = []
    for block in page.get_text("rawdict", clip=bbox)["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                x, y = line["dir"]
                text_angle = math.degrees(math.atan2(y, x))

                words = []
                for span in line["spans"]:
                    word_rect = pymupdf.Rect()
                    word_text = ""
                    for char in span["chars"]:
                        if char["c"] == " " and len(word_text) > 0:
                            words.append(TextWord(word_rect, word_text, page.number + 1))
                            word_text = ""
                            word_rect = pymupdf.Rect()
                        if char["c"] != " ":
                            word_text += char["c"]
                            word_rect.include_rect(pymupdf.Rect(char["bbox"]) * page.rotation_matrix)
                    if len(word_text) > 0:
                        words.append(TextWord(word_rect, word_text, page.number + 1))

                raw_lines.append(TextLine(words, text_angle))

    lines = []
    current_line_words = []
    for line_index, raw_line in enumerate(raw_lines):
        for word_index, word in enumerate(raw_line.words):
            remaining_line = TextLine(raw_line.words[word_index:])
            # Check if the remaining words of the line should be treated as a separate text line, even if they are
            # only a tailing segment of the "raw line" as it was extracted from the PDF.
            if len(current_line_words) > 0 and remaining_line.is_line_start(lines, raw_lines[line_index + 1 :]):
                lines.append(TextLine(current_line_words, raw_line.text_angle))
                current_line_words = []
            current_line_words.append(word)
        if current_line_words:
            lines.append(TextLine(current_line_words, raw_line.text_angle))
            current_line_words = []

    return lines
