"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from extraction.features.utils.data_extractor import ExtractedFeature, FeatureOnPage
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textline import TextLine

# TODO those in config
keywords = [
    "bohrung",
    "sondierung",
    "sondage",
    "baggerschlitz",
    "baggerschacht",
    "bohrstelle",
    "sondierschacht",
    "schachtprofil",
    "forage",
    "tranchée",
]

excluded_keywords = ["n.r", "n", "nº", "nr", "nummer", "no"]

MIN_VERTICAL_OVERLAP = 0.9


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


def _find_closest_nearby_line(current_line: TextLine, all_lines: list[TextLine]) -> TextLine | None:
    """Find the line that is the closest to the current line on the right.

    Args:
        current_line: The line containing the keyword.
        all_lines: All text lines from the document.

    Returns:
        TextLine | None: List of nearby lines that could contain the title.
    """
    nearby_lines = [
        line
        for line in all_lines
        if line.rect.x0 > current_line.rect.x1
        and y_overlap_significant_smallest(current_line.rect, line.rect, MIN_VERTICAL_OVERLAP)
    ]
    return min(nearby_lines, key=lambda line: line.rect.x0 - current_line.rect.x1) if nearby_lines else None


def _remove_any_keyword(text: str, keywords: list[str]) -> str | None:
    """Remove predefined keywords and normalize the given text.

    This function scans the input text for any of the provided keywords (case-insensitive),
    removes them, and performs basic cleanup by replacing punctuation and collapsing
    multiple spaces.

    Args:
        text (str): The input text to clean.
        keywords (list[str]): A list of keywords to remove from the text.

    Returns:
        str | None: The cleaned and normalized text or None if empty text.
    """
    # Build regex pattern for keywords (escaped and followed by a word boundary)
    pattern = "(" + r"\b" + "|".join(re.escape(kw) + r"\b" for kw in keywords) + ")"

    # Remove matched keywords (case-insensitive)
    cleaned = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove scale from text (eg: "1:100")
    cleaned = re.sub(r"1:\d{1,3}", " ", cleaned)

    # Replace punctuation, normalize whitespace and remove trailing spaces
    cleaned = re.sub(r"[:._]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    # TODO could check that the name != filename (avoid those cantonal ID, e.g. 5115.pdf, see issue description)

    # Check if result is empty
    if len(cleaned) == 0:
        return None

    return cleaned


def _match_any_keyword(text: str, keywords: list[str]) -> re.Match | None:
    """Search a text for the first occurrence of any keyword from a predefined list.

    The search is case-insensitive and all special characters in the keywords are
    properly escaped to ensure literal matching. Each keyword is treated as a whole word
    or as the ending of a longer word (e.g., "bohrung" matches "bohrung" or "sondierbohrung").
    If multiple keywords are present in the text, only the first match (in reading order)
    is returned.

    Args:
        text (str): The text to search within.
        keywords (list[str]): A list of keywords to look for.

    Returns:
        re.Match | None: A re.Match object for the first found keyword, or None
        if no match is found.
    """
    # Build a regex pattern that matches keywords
    pattern = "(" + "|".join(re.escape(kw) + r"\b" for kw in keywords) + ")"
    # Perform a case-insensitive search
    return re.search(pattern, text, re.IGNORECASE)


def extract_borehole_names(text_lines: list[TextLine]) -> list[FeatureOnPage[BoreholeName]]:
    """Extract borehole name from text lines.

    The borehole name can appear either:
    - In the same line as one of the keywords
    - In a nearby line to the right of a line containing a keyword

    Args:
        text_lines (list[TextLine]): List of TextLine objects to search through

    Returns:
        list[FeatureOnPage[BoreholeName]]: A list of extracted borehole names, if found
    """
    candidates: list[BoreholeName] = []

    for line in text_lines:
        # Check line for keyword
        match = _match_any_keyword(line.text, keywords)
        if not match:
            continue

        # Try same-line first
        same_line_name = line.text[match.end() :]
        if cleaned := _remove_any_keyword(same_line_name, excluded_keywords):
            candidates.append(
                FeatureOnPage(
                    feature=BoreholeName(name=cleaned, confidence=1.0),
                    rect=line.rect,
                    page=line.page_number,
                )
            )
            continue

        # Fallback: closest line to the right
        hit_line = _find_closest_nearby_line(line, text_lines)
        if not hit_line:
            continue

        # Confidence based on horizontal gap (non-negative)
        dx = max(0.0, hit_line.rect.x0 - line.rect.x1)
        confidence = 1.0 / (1.0 + dx)

        if cleaned := _remove_any_keyword(hit_line.text, excluded_keywords):
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
    candidates.sort(key=lambda bh_name: (bh_name.feature.confidence, -bh_name.rect.y0), reverse=True)
    unique_candidates = list(set(candidates))

    return unique_candidates
