"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from swissgeol_doc_processing.geometry.util import y_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from swissgeol_doc_processing.utils.data_extractor import ExtractedFeature, FeatureOnPage
from swissgeol_doc_processing.utils.language_filtering import normalize_spaces, remove_any_keyword


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

    # Replace punctuation, normalize whitespace and remove trailing spaces
    cleaned = re.sub(r"[:._]", " ", text)
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

    # only horizontal lines
    horizontal_lines = [line for line in text_lines if abs(line.text_angle) < 5]
    line_heights = sorted([line.rect.height for line in horizontal_lines])
    if len(line_heights) < 2:
        return []

    median_line_height = line_heights[len(line_heights) // 2]
    percentile_90_line_height = line_heights[int(0.9 * len(line_heights))]

    # Iterate over all lines
    for line in horizontal_lines:
        is_tall_line = line.rect.height > 1.5 * median_line_height and line.rect.height > percentile_90_line_height
        candidates.extend(_extract_borehole_names_from_line(line, text_lines, is_tall_line, name_detection_params))

    if not candidates:
        return []

    # Sort unique candidates by highest confidence and height on the page
    candidates.sort(key=lambda bh_name: (bh_name.feature.confidence, -bh_name.rect.y0), reverse=True)

    highest_confidence = candidates[0].feature.confidence
    # we expect all borehole names from the same page to have a similar confidence
    return [candidate for candidate in candidates if candidate.feature.confidence > 0.8 * highest_confidence]


def _extract_borehole_names_from_line(
    line: TextLine, lines_for_fallback: list[TextLine], is_tall_line: bool, name_detection_params: dict
) -> list[FeatureOnPage[BoreholeName]]:
    candidates: list[FeatureOnPage[BoreholeName]] = []

    keywords: list[str] = name_detection_params["matching_keywords"]
    excluded_keywords: list[str] = name_detection_params.get("excluded_keywords")
    min_vertical_overlap = name_detection_params.get("min_vertical_overlap", 1.0)
    max_horizontal_distance = name_detection_params.get("max_horizontal_distance", 1e16)

    words = line.words

    if len(words) == 0:
        return []

    # convert e.g. "1 : 100" to a single word "1:100" for more consistent behaviour
    contracted_words = []
    skip_next = False
    for index, word in enumerate(words):
        if word.text in {"-", ".", ":", "/"} and 0 < index < len(words) - 1:
            previous_word = words[index - 1]
            next_word = words[index + 1]
            if previous_word.text[-1].isalnum() and next_word.text[0].isalnum():
                new_rect = previous_word.rect | word.rect | next_word.rect
                new_text = previous_word.text + word.text + next_word.text
                new_word = TextWord(new_rect, new_text, word.page_number)
                contracted_words.insert(len(contracted_words) - 1, new_word)
                skip_next = True
        else:
            if skip_next:
                skip_next = False
            else:
                contracted_words.append(word)

    words = contracted_words

    first_word = words[0]
    if first_word.text.lower() in {"anhang", "allegato", "annexe"}:
        return []

    keyword_match_length = _keyword_match_length([word.text for word in words], keywords)
    prefix_is_keyword = keyword_match_length > 0

    words = words[keyword_match_length:]

    if len(lines_for_fallback) and all(_is_number_keyword(word.text, excluded_keywords) for word in words):
        # The current line only contains a matched keyword (e.g. "Bohrung") and/or a number keyword (e.g. "nr").
        # Fallback: closest line to the right. Concatenate the two lines and try again.
        if hit_line := _find_closest_nearby_line(
            line, lines_for_fallback, min_vertical_overlap, max_horizontal_distance
        ):
            concatenated_line = TextLine(line.words + hit_line.words, line.text_angle)
            return _extract_borehole_names_from_line(
                concatenated_line,
                lines_for_fallback=list(),
                is_tall_line=is_tall_line,
                name_detection_params=name_detection_params,
            )
        else:
            return []

    interrupt_index = None
    for index, word in enumerate(words):
        # detect a scale like "1:50" or an opening parenthesis
        if re.search(r"\d+:\d+", word.text) or "(" in word.text:
            interrupt_index = index
            break
    if interrupt_index is not None:
        words = words[:interrupt_index]

    allow_simple = prefix_is_keyword or is_tall_line
    for start, end in _find_candidate_names(
        [word.text for word in words], excluded_keywords, allow_simple=allow_simple
    ):
        name: str = " ".join([word.text for word in words[start:end]])

        rect = pymupdf.Rect()
        for word in words[start:end]:
            rect.include_rect(word.rect)

        has_number_prefix = start > 0 and _is_number_keyword(words[start - 1].text, excluded_keywords)

        is_at_start = start == 1 if has_number_prefix else start == 0
        is_at_end = end == len(words)
        has_lowercase = any(char.isalpha() and char.islower() for char in name)
        has_letter_and_number = any(char.isalpha() for char in name) and any(char.isdigit() for char in name)

        # if the name is not immediately following the prefix keywords, then don't treat it as something special
        if not is_at_start:
            prefix_is_keyword = False

        if not (
            prefix_is_keyword
            or has_number_prefix
            or (is_tall_line and (has_letter_and_number or (is_at_start and is_at_end)))
            or (has_letter_and_number and not has_lowercase and is_at_end)
        ):
            continue

        confidence = rect.height
        if prefix_is_keyword:
            confidence *= 3
        if has_number_prefix:
            confidence *= 2
        if is_at_start:
            confidence *= 1.5
        if is_at_end:
            confidence *= 1.5
        if has_lowercase:
            confidence *= 0.5
        if has_letter_and_number:
            confidence *= 1.2

        # Step 2: Clean detection
        if text_cleaned := clean_borehole_name(name, excluded_keywords):
            candidates.append(
                FeatureOnPage(
                    feature=BoreholeName(name=text_cleaned, confidence=confidence),
                    rect=rect,
                    page=line.page_number,
                )
            )
            continue
    return candidates


def _is_number_keyword(word: str, excluded_keywords: list[str]) -> bool:
    alnum_only = "".join(char for char in word if char.isalnum()).lower()
    return alnum_only in excluded_keywords


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


def _find_candidate_names(
    words: list[str], excluded_keywords: list[str], allow_simple: bool = False
) -> list[tuple[int, int]]:
    results = []

    start = None
    current_candidate_is_valid = False
    for index, word in enumerate(words):
        # (?![0-9_])\w  for matching any Unicode letter
        match_letter_before_number = re.search(r"(?![0-9_])\w.*\d", word)
        # require at most 6 digits to avoid matching with coordinate pairs
        match_number_symbol_number = re.search(r"\d[/\-]\d", word) and sum(1 for char in word if char.isdigit()) <= 7

        has_letter = any(char.isalpha() for char in word)
        has_digit = any(char.isdigit() for char in word)
        starts_with_digit = len(word) and word[0].isdigit()
        all_uppercase = has_letter and word.isupper()

        is_simple_match = has_digit or (len(word) == 1 and word.isalpha())
        is_number_keyword = _is_number_keyword(word, excluded_keywords)

        if not current_candidate_is_valid and not starts_with_digit:
            # first word not valid on its own and second word does not start with a digit -> start again
            start = None

        if start is None:
            if is_number_keyword:
                continue
            if (
                match_letter_before_number
                or match_number_symbol_number
                or (allow_simple and is_simple_match and len(word) <= 5)
                # simple match with length 5 allows e.g. "6056. 12" from Geoquat B537.pdf
            ):
                start = index
                current_candidate_is_valid = True
            if has_letter and not has_digit and (len(word) <= 3 or all_uppercase):
                start = index
        else:
            if is_number_keyword or not is_simple_match:
                results.append((start, index))
                start = None
                current_candidate_is_valid = False
            else:
                current_candidate_is_valid = current_candidate_is_valid or has_digit
    if start is not None and current_candidate_is_valid:
        results.append((start, len(words)))

    return results
