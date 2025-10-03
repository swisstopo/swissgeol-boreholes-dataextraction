"""Model for the extraction of the name of a borehole."""

from __future__ import annotations

import re

import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textline import TextLine

keywords = ["bohrung", "sondierung", "sondage", "Sondierbohrung"]  # TODO those in config

excluded_keywords = ["N", "Nº", "Nr", "Nummer"]  # TODO those in config

MIN_VERTICAL_OVERLAP = 0.9


class BoreholeName(RectWithPageMixin):
    """Class to hold the name of a borehole."""

    def __init__(self, name: str, confidence: float, rect: pymupdf.Rect, page: int):
        self.name = name
        self.confidence = confidence
        self.rect_with_page = RectWithPage(rect, page)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}"

    def __hash__(self) -> int:
        """Make BoreholeName hashable for use in sets.

        Returns:
            Hash based on name, page and rect coordinates
        """
        return hash((self.name, self.page_number, self.rect))


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


def extract_borehole_names(text_lines: list[TextLine]) -> str | None:
    """Extract borehole name from text lines.

    The borehole name can appear either:
    - In the same line as one of the keywords
    - In a nearby line to the right of a line containing a keyword

    Args:
        text_lines: List of TextLine objects to search through

    Returns:
        The extracted borehole name if found, None otherwise
    """
    pattern = "(" + "|".join(re.escape(kw) + r"\b" for kw in keywords) + ")"
    candidates: list[BoreholeName] = []  # [(name, confidence), ...]

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
        candidates.append(BoreholeName(cleaned_name, confidence, hit_line.rect, hit_line.page_number))

    if not candidates:
        return None

    # Return the candidate with highest confidence
    candidates.sort(key=lambda bh_name: (bh_name.confidence, -bh_name.rect.y0), reverse=True)
    unique_candidates = list(set(candidates))
    return unique_candidates  # TODO then each should be mapped to the clossest borehole, like elevations for example.
