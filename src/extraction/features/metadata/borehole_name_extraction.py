"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re
from dataclasses import dataclass

from extraction.features.utils.data_extractor import ExtractedFeature, FeatureOnPage
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textline import TextLine

keywords = ["bohrung", "sondierung", "sondage", "Sondierbohrung"]  # TODO those in config

excluded_keywords = ["N", "Nº", "Nr", "Nummer"]  # TODO those in config

MIN_VERTICAL_OVERLAP = 0.9


@dataclass
class BoreholeName(ExtractedFeature):
    """Abstract class for Name Information."""

    name: str  # Name of the borhole
    confidence: float

    def __post_init__(self):
        """Checks if the information is valid."""
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string")
        if not isinstance(self.name, str):
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
            Name: The borhole's name information object.
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


def _find_clossest_nearby_line(current_line: TextLine, all_lines: list[TextLine]) -> TextLine | None:
    """Find the line that is the closest to the current line on the right.

    Args:
        current_line: The line containing the keyword
        all_lines: All text lines from the document

    Returns:
        List of nearby lines that could contain the title
    """
    nearby_lines = [
        line
        for line in all_lines
        if line.rect.x0 > current_line.rect.x1
        and y_overlap_significant_smallest(current_line.rect, line.rect, MIN_VERTICAL_OVERLAP)
        and _clean_name(line.text)
    ]
    return min(nearby_lines, key=lambda line: line.rect.x0 - current_line.rect.x1) if nearby_lines else None


def _clean_name(text: str) -> str:
    """Clean and normalize the extracted name.

    Args:
        text: Text to clean
        pattern: Regex pattern for keywords to remove

    Returns:
        The cleaned text after the keyword
    """
    pattern = "(" + "|".join(re.escape(kw) + r"\b" for kw in keywords) + ")"
    match = re.search(pattern, text, re.IGNORECASE)
    text_after = text[match.end() :] if match else text

    excluded_pattern = "(" + "|".join(re.escape(kw) + r"\b" for kw in excluded_keywords) + ")"
    cleaned = re.sub(excluded_pattern, " ", text_after)
    cleaned = re.sub(r"[:\._]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # TODO could check that the name != filename (avoid those cantonal ID, e.g. 5115.pdf, see issue description)
    return cleaned.strip()


def extract_borehole_names(text_lines: list[TextLine]) -> list[FeatureOnPage[BoreholeName]]:
    """Extract borehole name from text lines.

    The borehole name can appear either:
    - In the same line as one of the keywords
    - In a nearby line to the right of a line containing a keyword

    Args:
        text_lines: List of TextLine objects to search through

    Returns:
        A list of extracted borehole names, if found
    """
    pattern = "(" + "|".join(re.escape(kw) + r"\b" for kw in keywords) + ")"
    candidates: list[BoreholeName] = []

    for current_line in text_lines:
        match = re.search(pattern, current_line.text, re.IGNORECASE)
        if not match:
            continue

        cleaned_name = _clean_name(current_line.text)
        if cleaned_name:
            # Case 1: Name in the same line after the keyword
            hit_line = current_line
            confidence = 1.0
        else:
            # Case 2: Current line contains only the keyword, look for name in lines to the right
            hit_line = _find_clossest_nearby_line(current_line, text_lines)
            if not hit_line:
                continue
            confidence = 1 / (1 + hit_line.rect.x0 - current_line.rect.x1)
            cleaned_name = _clean_name(hit_line.text)

        # Define name feature
        candidates.append(
            FeatureOnPage(
                feature=BoreholeName(name=cleaned_name, confidence=confidence),
                rect=hit_line.rect,
                page=hit_line.page_number,
            )
        )

    if not candidates:
        return []

    # Sort the candidates by highest confidence and height on the page
    candidates.sort(key=lambda bh_name: (bh_name.feature.confidence, -bh_name.rect.y0), reverse=True)
    unique_candidates = list(set(candidates))

    return unique_candidates
