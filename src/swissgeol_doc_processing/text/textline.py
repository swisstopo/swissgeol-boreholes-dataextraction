"""This module contains utility functions and classes for TextLine objects."""

from __future__ import annotations

import re

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.text.stemmer import find_matching_expressions


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

    def __init__(self, words: list[TextWord], text_angle: float = 0):
        """Initialize the TextLine object.

        Args:
            words (list[TextWord]): The words that make up the line.
            text_angle (float): The angle in degrees (between -180 and 180) that the text makes with the x-axis.
        """
        rect = rect_union([word.rect for word in words])
        self.rect_with_page = RectWithPage(rect, next((word.page_number for word in words), None))
        self.words = words
        self.text_angle = text_angle
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

        Uses stemming to handle word variations across german, french, english and italian and
        additionally compound split in case of german.

        Args:
            parameters (dict): The parameter dictionary containing the used expressions and thresholds.
            language (str): The language of the material description, e.g. "de", "fr", "en", "it".
            analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
            search_excluding (bool): If True, search for excluding expressions, otherwise for including expressions.

        Returns:
            bool: True if the line contains any of the material description expressions, False otherwise.
        """
        # Tokenize and stem words in the text
        text_tokens = re.findall(r"\b\w+\b", self.text)
        exp_type = "including_expressions" if not search_excluding else "excluding_expressions"
        patterns = parameters["material_description"][language][exp_type]
        split_threshold = parameters.get("compound_split_threshold", 0.4)

        return find_matching_expressions(patterns, split_threshold, text_tokens, language, analytics, search_excluding)

    def to_json(self) -> dict:
        """Convert the TextLine object to a JSON serializable dictionary."""
        return {
            "text": self.text,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.page_number,
        }


def rect_union(char_rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    """Takes the union of all the rects in the list.

    Contrary to the pymupdf methods include_rect and |, this implementation also works well for zero-width or
    zero-height rects.
    """
    if len(char_rects) == 0:
        return pymupdf.Rect()
    else:
        return pymupdf.Rect(
            min(rect.x0 for rect in char_rects),
            min(rect.y0 for rect in char_rects),
            max(rect.x1 for rect in char_rects),
            max(rect.y1 for rect in char_rects),
        )
