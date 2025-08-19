"""This module contains utility functions and classes for TextLine objects."""

from __future__ import annotations

import re

import pymupdf
from compound_split import char_split
from nltk.stem.snowball import SnowballStemmer

from extraction.features.utils.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from extraction.features.utils.geometry.util import x_overlap_significant_largest
from extraction.features.utils.text.matching_params_analytics import MatchingParamsAnalytics, track_match


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

    A TextLine object is a collection of DepthInterval objects.
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
        self.rect_with_page = RectWithPage(rect, words[0].page_number)
        self.words = words
        self.stemmers = {}
        self.stemmer_languages = {"de": "german", "fr": "french", "en": "english", "it": "italian"}

    def _get_stemmer(self, language: str) -> SnowballStemmer:
        """Get the appropriate stemmer for the given language.

        Args:
            language (str): The language for which to get the stemmer, e.g. "de", "fr", "en", "it".

        Returns:
            SnowballStemmer: The stemmer for the specified language.
        """
        # Create appropriate stemmer based on language
        if language not in self.stemmers:
            stemmer_lang = self.stemmer_languages.get(language, "german")
            self.stemmers[language] = SnowballStemmer(stemmer_lang)
        return self.stemmers[language]

    def _split_compounds(self, tokens: list[str], split_threshold: float) -> list[str]:
        """Split compound words using char_split and return processed list.

        This method uses  an ngram-based compound splitter for German language based on
        Tuggener, Don (2016):  https://pypi.org/project/compound-split/

        Args:
            tokens (List[str]): List of tokens to process.
            split_threshold (float): Threshold for splitting compounds

        Returns:
            List[str]: Processed list of tokens with compounds split.
        """
        processed_tokens = []
        for token in tokens:
            comp_split = char_split.split_compound(token)[0]
            if comp_split[0] > split_threshold:
                processed_tokens.extend(comp_split[1:])
            else:
                processed_tokens.append(token)

        return processed_tokens

    def _find_matching_expressions(
        self,
        patterns: list,
        split_threshold: float,
        targets: list[str],
        language: str,
        analytics: MatchingParamsAnalytics | None = None,
        search_excluding: bool = False,
    ) -> bool:
        """Check if any of the patterns match the targets for german use a second check against compound split.

        Args:
            patterns (List): A list of patterns to match against.
            split_threshold (float): Threshold for splitting compounds.
            targets (List[str]): A list of target strings to match against.
            language (str): The language of the patterns, used for stemming.
            analytics ([MatchingParamsAnalytics]): Aanalytics instance to track matches.
            search_excluding (bool): Whether this is for excluding expressions (for analytics).

        Returns:
            bool: True if any pattern matches, False otherwise.
        """
        stemmer = self._get_stemmer(language)

        patterns = [stemmer.stem(p.lower()) for p in patterns]
        targets = {stemmer.stem(t.lower()) for t in targets}

        if language == "de" and not search_excluding:
            targets_split = self._split_compounds(targets, split_threshold)
            targets_split = {stemmer.stem(t.lower()) for t in targets_split}

            targets_to_check = targets | targets_split
        else:
            targets_to_check = targets

        for item in patterns:
            if item in targets_to_check:
                track_match(analytics, item, language, search_excluding)
                return True

        return False

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

        return self._find_matching_expressions(
            patterns, split_threshold, text_tokens, language, analytics, search_excluding
        )

    @property
    def text(self) -> str:
        """Get the text of the line."""
        return " ".join([word.text for word in self.words])

    def __repr__(self) -> str:
        return f"TextLine({self.text}, {self.rect})"

    """
    Check if the current line can be trusted as a stand-alone line, even if it is only a tailing segment of a line that
    was directly extracted from the PDF. This decision is made based on the location (especially x0-coordinates) of the
    lines above and below. If there are enough lines with matching x0-coordinates, then we can assume that this lines
    also belongs to the same "column" in the page layout. This is necessary, because text extraction from PDF sometimes
    extracts text lines too "inclusively", resulting in lines that span across different columns.

    The logic is still not very robust. A more robust solution will be possible once we include line detection as a
    feature in this pipeline as well.
    """

    def is_line_start(self, raw_lines_before: list[TextLine], raw_lines_after: list[TextLine]) -> bool:
        """Check if the current line is the start of a new line."""

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
