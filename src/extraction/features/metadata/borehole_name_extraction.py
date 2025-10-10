"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from extraction.features.utils.data_extractor import ExtractedFeature, FeatureOnPage
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textline import TextLine
from utils.language_filtering import match_any_keyword, normalize_spaces, remove_any_keyword


@dataclass
class BoreholeName(ExtractedFeature):
    """Abstract class for Name Information."""

    name: str  # Name of the borehole
    confidence: float  # Confidence score based on distance

    def __post_init__(self):
        """Checks if the information is valid."""
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string")
        if not isinstance(self.confidence, float):
            raise ValueError("Confidence must be a float")

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
        return {"name": self.name, "confidence": self.confidence}

    @classmethod
    def from_json(cls, data: dict) -> BoreholeName:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the elevation information.

        Returns:
            BoreholeName: The borehole's name information object.
        """
        return cls(name=data["name"], confidence=data["confidence"])


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
    current_line: TextLine, all_lines: list[TextLine], min_vertical_overlap: float
) -> TextLine | None:
    """Find the line that is the closest to the current line on the right.

    Args:
        current_line (TextLine): The line containing the keyword.
        all_lines (list[TextLine]): All text lines from the document.
        min_vertical_overlap (float): Overlap threshold for closest line detection.

    Returns:
        TextLine | None: List of nearby lines that could contain the title.
    """
    nearby_lines = [
        line
        for line in all_lines
        if line.rect.x0 > current_line.rect.x1
        and y_overlap_significant_smallest(current_line.rect, line.rect, min_vertical_overlap)
    ]
    return min(nearby_lines, key=lambda line: line.rect.x0 - current_line.rect.x1) if nearby_lines else None


def _clean_borehole_name(text: str, excluded_keywords: list[str]) -> str | None:
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
    cleaned = re.sub(r"1:\d{1,3}", " ", text)
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)

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

    The algorithm scans each line for any `matching_keywords` at the end of a word
    (to avoid plural/inflected forms). If a keyword is found:
    - **Same-line extraction:** take the substring **after** the match on the same line; clean it
        by removing any `excluded_keywords`. If a non-empty name remains, emit a candidate with
        confidence = 1.0 and the line’s bounding box.
    - **Right-side fallback:** if same-line fails, find the closest line to the **right** that
        vertically overlaps by at least `min_vertical_overlap`. Compute confidence as
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
    matching_keywords = name_detection_params.get("matching_keywords", [])
    excluded_keywords = name_detection_params.get("excluded_keywords", [])
    min_vertical_overlap = name_detection_params.get("min_vertical_overlap", 1.0)

    for line in text_lines:
        # Check line for keyword - Enforce end to avoid plural form
        match = match_any_keyword(line.text, matching_keywords, end=True)
        if not match:
            continue

        # Try same-line first
        same_line_name = line.text[match.end() :]
        if cleaned := _clean_borehole_name(same_line_name, excluded_keywords):
            candidates.append(
                FeatureOnPage(
                    feature=BoreholeName(name=cleaned, confidence=1.0),
                    rect=line.rect,
                    page=line.page_number,
                )
            )
            continue

        # Fallback: closest line to the right
        hit_line = _find_closest_nearby_line(line, text_lines, min_vertical_overlap)
        if not hit_line:
            continue

        # Confidence based on horizontal gap (non-negative)
        dy = max(0.0, hit_line.rect.y1 - hit_line.rect.y0)
        dx = max(0.0, hit_line.rect.x0 - line.rect.x1)
        confidence = dy / (1 + dy + dx)

        if cleaned := _clean_borehole_name(hit_line.text, excluded_keywords):
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

    # Sort the candidates by highest confidence and height on the page
    # TODO Use confidence for better matching
    candidates.sort(key=lambda bh_name: (bh_name.feature.confidence, -bh_name.rect.y0), reverse=True)
    unique_candidates = list(set(candidates))

    return unique_candidates
