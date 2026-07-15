"""Methods for extracting plain text from a PDF document.

Run with
uv run python -m scripts.extract_full_text_csv -i data/pdfs/ -o full_text.json

e.g.
uv run python -m scripts.extract_full_text_csv -i data/zurich/ -o data/filteredtext/zurich_full_text.json --header-only

"""

import re

import pymupdf

from swissgeol_doc_processing.text.textline import TextLine, TextWord

NUMBER_PATTERN = re.compile(r"^-?\d+([.,]\d+)?$")
PUNCTUATION_PATTERN = re.compile(r"^[?%!><.,/\\-]+$=")


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
            # Check if the remaining words of the line should be treated as a separate text line, even if they are
            # only a tailing segment of the "raw line" as it was extracted from the PDF.
            if len(current_line_words) > 0 and remaining_line.is_line_start(lines, raw_lines[line_index + 1 :]):
                lines.append(TextLine(current_line_words))
                current_line_words = []
            current_line_words.append(word)
        if current_line_words:
            lines.append(TextLine(current_line_words))
            current_line_words = []

    return lines


def filter_header_candidate_lines(lines: list[TextLine], language: str, matching_params: dict) -> list[TextLine]:
    """Remove material description lines, standalone numbers and standalone punctuation from a list of lines.

    Intended for use cases (e.g. document-level classification) where only the "header-like" remainder of
    the page text is of interest, since material descriptions, numeric depth/coordinate values and stray
    punctuation marks make up most of a borehole log's token count without being relevant header content.

    Args:
        lines (list[TextLine]): The text lines to filter, e.g. as returned by `extract_text_lines`.
        language (str): The language of the document, e.g. "de", "fr", "en", "it".
        matching_params (dict): The matching parameters, as used by `TextLine.is_description`.

    Returns:
        list[TextLine]: The filtered text lines, with material description lines removed entirely and
            standalone number/punctuation words removed from the remaining lines.
    """
    header_lines = []
    for line in lines:
        if line.is_description(matching_params, language):
            continue
        remaining_words = [
            word
            for word in line.words
            if not NUMBER_PATTERN.match(word.text) and not PUNCTUATION_PATTERN.match(word.text)
        ]
        if remaining_words:
            header_lines.append(TextLine(remaining_words))
    return header_lines
