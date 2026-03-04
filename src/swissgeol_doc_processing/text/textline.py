"""This module contains utility functions and classes for TextLine objects."""

from __future__ import annotations

import re
import unicodedata

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from swissgeol_doc_processing.geometry.util import x_overlap_significant_largest
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.text.stemmer import find_matching_expressions

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Collapse common OCR confusions to a canonical representation.
_CONFUSION_MAP = str.maketrans(
    {
        "m": "n",
        "n": "n",
        "o": "o",
        "a": "o",
        "i": "i",
        "l": "i",
        "e": "e",
        "c": "e",
    }
)


def _norm(s: str) -> str:
    """Normalize a string for matching.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string (lowercased, diacritics removed).
    """
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    return s


def _confusion_norm(s: str) -> str:
    """Normalize a string and collapse common OCR confusions.

    This is a conservative OCR-specific normalization step. It intentionally loses
    information (e.g. 'm'/'n') so that OCR variants can match the same target.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string with confusion groups collapsed.
    """
    return _norm(s).translate(_CONFUSION_MAP)


def contains_expression(
    text: str,
    expr: str,
    *,
    min_token_length: int = 6,
) -> bool:
    """Check whether `expr` occurs in `text`, optionally allowing OCR/typo tolerance.

    Args:
        text (str): Text to search in.
        expr (str): Expression/pattern to search for.
        min_token_length (int): Minimum token length for tolerance-based matching.
            Tokens shorter than this will not be matched fuzzily (to reduce false positives).

    Returns:
        bool: True if the expression is considered present in the text, False otherwise.
    """
    t = _confusion_norm(text)
    e = _confusion_norm(expr)

    # exact substring on normalized text
    if len(e) < min_token_length:
        return False
    return e in t


class TextWord(RectWithPageMixin):
    """Class to represent a word on a specific location on a PDF page.

    A TextWord object consists of a pymupdf Rectangle object and a string.
    The string is the word that is contained in the rectangle. The rectangles are used
    to represent the location of the word in a PDF document.
    """

    def __init__(self, rect: pymupdf.Rect, text: str, page: int):
        self.rect_with_page = RectWithPage(rect, page)
        self.text = text

    def __repr__(self) -> str:
        return f"TextWord({self.rect}, {self.text})"


class TextLine(RectWithPageMixin):
    """Class to represent TextLine objects.

    A TextLine object is a collection of TextWord objects.
    It is used to represent a line of text in a PDF document.
    """

    def __init__(self, words: list[TextWord]):
        """Initialize the TextLine object.

        Args:
            words (list[TextWord]): The words that make up the line.
            page_number (int): The page number of the line. The first page has idx 1.
        """
        rect = pymupdf.Rect()
        for word in words:
            rect.include_rect(word.rect)
        self.rect_with_page = RectWithPage(rect, next((word.page_number for word in words), None))
        self.words = words
        self.is_indented = False

    def __repr__(self) -> str:
        return f"TextLine({self.text}, {self.rect})"

    @property
    def text(self) -> str:
        """Get the text of the line."""
        return " ".join([word.text for word in self.words])

    def is_description(
        self,
        parameters: dict,
        language: str,
        analytics: MatchingParamsAnalytics | None = None,
        search_excluding: bool = False,
    ) -> bool:
        """Check if the line is a material description.

        Strategy:
        1) Use stemming/compound-split matching.
        2) Optionally apply a conservative OCR-confusion fallback (m/n, o/a, i/l, e/c).


        Args:
            parameters (dict): The parameter dictionary containing the used expressions and thresholds.
            language (str): The language of the material description, e.g. "de", "fr", "en", "it".
            analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
            search_excluding (bool): If True, search for excluding expressions, otherwise for including expressions.

        Returns:
            bool: True if the line contains any of the material description expressions, False otherwise.
        """
        text = self.text
        text_tokens = re.findall(r"\b\w+\b", text)

        exp_type = "excluding_expressions" if search_excluding else "including_expressions"
        patterns = parameters["material_description"][language][exp_type]
        split_threshold = parameters.get("compound_split_threshold", 0.4)

        # 1) stemming matching
        if find_matching_expressions(patterns, split_threshold, text_tokens, language, analytics, search_excluding):
            return True

        # 2) # OCR confusion fallback
        typo_cfg = parameters.get("material_description", {}).get("typo_tolerance", {})
        if not typo_cfg.get("enabled", False):
            return False

        # only apply to tokens >= 6 to avoid matching short tokens too easily.
        confusion_enabled = typo_cfg.get("confusion_enabled", True)
        confusion_min_len = int(typo_cfg.get("confusion_min_token_length", 6))

        if confusion_enabled:
            for expr in patterns:
                if contains_expression(text, expr, min_token_length=confusion_min_len):
                    if analytics:
                        analytics.track_expression_match(expr, language, is_excluding=search_excluding)
                    return True

    def is_line_start(self, raw_lines_before: list[TextLine], raw_lines_after: list[TextLine]) -> bool:
        """Determine whether this line can be considered the start of a properly aligned text block.

        This method checks whether the line's x0-coordinate (left margin) aligns with surrounding
        lines above and below. If enough nearby lines share the same left margin (within a tolerance),
        the current line is assumed to belong to the same column and can be trusted as the start
        of a valid text segment.

        The check is necessary because PDF text extraction sometimes merges or splits lines in a way
        that spans multiple columns. This heuristic helps ensure column consistency, though it is
        not fully robust until line detection is integrated more directly.

        Args:
            raw_lines_before (list[TextLine]): Neighboring lines before this line.
            raw_lines_after (list[TextLine]): Neighboring lines after this line.

        Returns:
            bool: True if the line aligns consistently with its neighbors and can be trusted
                as the start of a column, False otherwise.
        """

        def significant_overlap(line: TextLine) -> bool:
            return x_overlap_significant_largest(line.rect, self.rect, 0.5)

        matching_lines_before = [line for line in raw_lines_before if significant_overlap(line)]
        matching_lines_after = [line for line in raw_lines_after if significant_overlap(line)]

        def count_points(lines: list[TextLine]) -> tuple[int, int]:
            exact_points = 0
            indentation_points = 0
            for other in lines:
                line_height = self.rect.height
                if max(other.rect.y0 - self.rect.y1, self.rect.y0 - other.rect.y1) > 5 * line_height:
                    # too far away vertically
                    return exact_points, indentation_points

                if abs(other.rect.x0 - self.rect.x0) < 0.2 * line_height:
                    exact_points += 1
                elif 0 < other.rect.x0 - self.rect.x0 < 2 * line_height:
                    indentation_points += 1
                else:
                    # other line is more to the left, and significantly more to the right (allowing for indentation)
                    return exact_points, indentation_points
            return exact_points, indentation_points

        # three lines before and three lines after
        exact_points_1, indentation_points_1 = count_points(matching_lines_before[:-4:-1])
        exact_points_2, indentation_points_2 = count_points(matching_lines_after[:3])
        exact_points = exact_points_1 + exact_points_2
        indentation_points = indentation_points_1 + indentation_points_2

        return exact_points >= 3 or (exact_points >= 2 and indentation_points >= 1)

    def to_json(self):
        """Convert the TextLine object to a JSON serializable dictionary."""
        return {
            "text": self.text,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.page_number,
        }
