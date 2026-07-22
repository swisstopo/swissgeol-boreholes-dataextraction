"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from swissgeol_doc_processing.geometry.util import y_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.data_extractor import ExtractedFeature, FeatureOnPage
from swissgeol_doc_processing.utils.language_filtering import (
    normalize_spaces,
    remove_any_keyword,
    remove_in_parenthesis,
    remove_scale,
)


@dataclass
class BoreholeName(ExtractedFeature):
    """Abstract class for Name Information."""

    name: str  # Name of the borehole
    confidence: float  # Confidence score based on distance

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Name(name={self.name}, confidence={self.confidence})"

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "name": self.name,
            "confidence": self.confidence,
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_json(cls, data: dict) -> BoreholeName:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the name information.

        Returns:
            BoreholeName: The borehole's name information object.
        """
        return cls(name=data["name"], confidence=data["confidence"], is_correct=data.get("is_correct"))


@dataclass
class NameInDocument:
    """Class for extracted borehole name information from a document."""

    name_feature_list: list[FeatureOnPage[BoreholeName]]
    filename: str

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.name_feature_list]


def _find_closest_nearby_line(
    current_line: TextLine, all_lines: list[TextLine], min_vertical_overlap: float, max_horizontal_distance: float
) -> TextLine | None:
    """Find the line that is the closest to the current line on the right.

    Args:
        current_line (TextLine): The line containing the keyword.
        all_lines (list[TextLine]): All text lines from the document.
        min_vertical_overlap (float): Overlap threshold for closest line detection.
        max_horizontal_distance (float): Maximal distance to look for next text.

    Returns:
        TextLine | None: List of nearby lines that could contain the name.
    """
    nearby_lines = [
        line
        for line in all_lines
        if current_line.rect.x1 < line.rect.x0 < current_line.rect.x1 + max_horizontal_distance
        and y_overlap_significant_smallest(current_line.rect, line.rect, min_vertical_overlap)
    ]
    return min(nearby_lines, key=lambda line: line.rect.x0 - current_line.rect.x1) if nearby_lines else None


def clean_borehole_name(text: str, excluded_keywords: list[str]) -> str | None:
    """Clean borehole name and normalize the given text.

    This function scans the input text for any of the provided keywords (case-insensitive),
    removes them, and performs basic cleanup by replacing punctuation and collapsing
    multiple spaces.

    Args:
        text (str): The input text to clean.
        excluded_keywords (list[str]): A list of keywords to remove from the text.

    Returns:
        str | None: The cleaned and normalized text or None if empty text.
    """
    # Remove matched keywords (case-insensitive)
    if excluded_keywords is not None and len(excluded_keywords) != 0:
        text = remove_any_keyword(text, excluded_keywords)

    # Remove scale from text (eg: "1:100")
    cleaned = remove_scale(text)
    cleaned = remove_in_parenthesis(cleaned)

    # Replace punctuation, normalize whitespace and remove trailing spaces
    cleaned = re.sub(r"[:._]", " ", cleaned)
    cleaned = normalize_spaces(cleaned)

    # Check if result is empty
    if len(cleaned) == 0:
        return None

    return cleaned


def extract_borehole_names(
    text_lines: list[TextLine], name_detection_params: dict
) -> list[FeatureOnPage[BoreholeName]]:
    """Extract borehole names from text lines using keyword anchors and a right-side fallback.

    The algorithm scans each line using two keyword lists:

    - **matching_keywords_suffix** (soft): matched at the end of a word (e.g. "bohrung" matches
        "kernbohrung"). The name is the substring after the match. The keyword itself is **not**
        included in the output.
    - **matching_keywords_inner** (strict): matched at both the **start** and **end** of a word.
        The matched keyword is included as a prefix in the output name (e.g. "KB 12").

    If either keyword type is found on a line:

    - **Same-line extraction:** clean the substring after the match by removing `excluded_keywords`.
        If a non-empty name remains, emit a candidate with confidence = 1.0 and the line’s bounding
        box.
    - **Right-side fallback:** if same-line extraction fails, find the closest line to the right
        that vertically overlaps by at least `min_vertical_overlap`. Compute confidence as
        `dy / (1 + dy + dx)` where `dy` is the right-line height and `dx` is the horizontal gap.
        If a cleaned name is found there, emit a candidate whose bounding box is the union of the
        anchor line and the right-side line.

    Args:
        text_lines (list[TextLine]): List of TextLine objects to search through
        name_detection_params (dict): The parameters for the name detection algorithm.

    Returns:
        list[FeatureOnPage[BoreholeName]]: A list of extracted borehole names, if found
    """
    candidates: list[FeatureOnPage[BoreholeName]] = []
    keywords = name_detection_params["matching_keywords"]
    excluded_keywords = name_detection_params.get("excluded_keywords", [])
    min_vertical_overlap = name_detection_params.get("min_vertical_overlap", 1.0)
    max_horizontal_distance = name_detection_params.get("max_horizontal_distance", 1e16)

    # only horizontal lines
    horizontal_lines = [line for line in text_lines if line.rect.width > line.rect.height]
    line_heights = sorted([line.rect.height for line in horizontal_lines])
    if len(line_heights) < 2:
        return []

    median_line_height = line_heights[len(line_heights) // 2 + 1]
    percentile_90_line_height = line_heights[int(0.9 * len(line_heights))]

    # Iterate over all lines
    for line in horizontal_lines:
        is_tall_line = line.rect.height > 1.25 * median_line_height and line.rect.height > percentile_90_line_height
        words = line.words

        if len(words) == 0:
            continue

        first_word = words[0]
        if first_word.text.lower() in {"anhang", "allegato", "annexe"}:
            continue

        keyword_match_length = _keyword_match_length([word.text for word in words], keywords)
        prefix_is_keyword = keyword_match_length > 0
        words = words[keyword_match_length:]

        scale_index = None
        for index, word in enumerate(words):
            # detect a scale like "1:50"
            if re.search(r"\d+:\d+", word.text):
                scale_index = index
                break
        if scale_index is not None:
            words = words[:scale_index]

        def is_excluded(word: str) -> bool:
            letter_only = "".join(char for char in word if char.isalpha()).lower()
            return letter_only in excluded_keywords

        words = [word for word in words if not is_excluded(word.text)]

        # Letter- or number-only borehole name as the only word on a line after a keyword prefix
        if prefix_is_keyword and len(words) == 1:
            word = words[0].text
            if (word.isalpha() and len(word) <= 1) or word.isdigit():
                candidates.append(
                    FeatureOnPage(
                        feature=BoreholeName(name=word, confidence=1),
                        rect=words[0].rect,
                        page=line.page_number,
                    )
                )
                continue

        if name_match := _find_candidate_name([word.text for word in words]):
            start, end = name_match
            name: str = " ".join([word.text for word in words[start:end]])

            rect = pymupdf.Rect()
            for word in words[start:end]:
                rect.include_rect(word.rect)

            is_at_start = start == 0
            is_at_end = end == len(words)

            if not prefix_is_keyword and not is_tall_line:
                continue

            confidence = rect.height
            if prefix_is_keyword:
                confidence *= 3
            if is_at_start:
                confidence *= 1.5
            if is_at_end:
                confidence *= 1.5

            # Step 2: Clean detection
            if text_cleaned := clean_borehole_name(name, excluded_keywords):
                print(name, text_cleaned)
                candidates.append(
                    FeatureOnPage(
                        feature=BoreholeName(name=text_cleaned, confidence=confidence),
                        rect=rect,
                        page=line.page_number,
                    )
                )
                continue

            # Fallback: closest line to the right
            hit_line = _find_closest_nearby_line(line, text_lines, min_vertical_overlap, max_horizontal_distance)
            if not hit_line:
                continue

            # Confidence based on horizontal gap (non-negative)
            dy = max(0.0, hit_line.rect.y1 - hit_line.rect.y0)
            dx = max(0.0, hit_line.rect.x0 - line.rect.x1)
            confidence = dy / (1 + dy + dx)

            if cleaned := clean_borehole_name(hit_line.text, excluded_keywords):
                # Define new bounding box as merge of both
                candidates.append(
                    FeatureOnPage(
                        feature=BoreholeName(name=cleaned, confidence=confidence),
                        rect=pymupdf.Rect(
                            min(line.rect.x0, hit_line.rect.x0),
                            min(line.rect.y0, hit_line.rect.y0),
                            max(line.rect.x1, hit_line.rect.x1),
                            max(line.rect.y1, hit_line.rect.y1),
                        ),
                        page=line.page_number,
                    )
                )

    if not candidates:
        return []

    # Sort unique candidates by highest confidence and height on the page
    # TODO Use confidence for better matching
    candidates.sort(key=lambda bh_name: (bh_name.feature.confidence, -bh_name.rect.y0), reverse=True)

    highest_confidence = candidates[0].feature.confidence
    # we expect all borehole names from the same page to have a similar confidence
    return [candidate for candidate in candidates if candidate.feature.confidence > 0.8 * highest_confidence]


def _preprocess_word(word: str) -> str:
    return "".join(char for char in word.lower() if char.isalnum())


def _keyword_match_length(words: list[str], keywords: list[str]) -> int:
    longest_match = 0
    keywords_preprocessed = [[_preprocess_word(word) for word in keyword.split(" ")] for keyword in keywords]
    words_preprocessed = [_preprocess_word(word) for word in words]
    for keyword in keywords_preprocessed:
        if len(keyword) > longest_match and words_preprocessed[: len(keyword)] == keyword:
            longest_match = len(keyword)
    return longest_match


def _find_candidate_name(words: list[str]) -> tuple[int, int] | None:
    start = None
    current_candidate_has_digit = False
    for index, word in enumerate(words):
        # (?![0-9_])\w  for matching any Unicode letter
        match_letter_before_number = re.search(r"(?![0-9_])\w.*\d", word)
        match_number_hyphen_number = re.search(r"\d-\d", word)

        has_letter = any(char.isalpha() for char in word)
        has_digit = any(char.isdigit() for char in word)
        starts_with_digit = len(word) and word[0].isdigit()
        all_uppercase = has_letter and word.isupper()

        if not current_candidate_has_digit and not starts_with_digit:
            # no digit in the first words and second word does not start with a digit -> start again
            start = None

        if start is None:
            if match_letter_before_number or match_number_hyphen_number:
                start = index
                current_candidate_has_digit = True
            if has_letter and not has_digit and (len(word) <= 3 or all_uppercase):
                start = index
        else:
            if has_digit or (len(word) == 1 and word.isalpha()):
                current_candidate_has_digit = current_candidate_has_digit or has_digit
                continue
            return start, index
    if start is not None and current_candidate_has_digit:
        return start, len(words)

    return None
