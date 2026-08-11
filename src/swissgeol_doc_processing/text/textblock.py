"""This module contains the TextBlock class, which represents a block of text in a PDF document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.data_extractor import (
    ExtractedFeature,
    FeatureOnPage,
)

# fixed list of German abbreviations spotted in practice, not exhaustive; add more as you find them.
_ABBREVIATIONS_ENDING_IN_PERIOD = ("max.", "z.T.", "bzw.", "ca.", "etc.", "z.B.")


@dataclass
class MaterialDescriptionLine(ExtractedFeature):
    """Class to represent a line of a material description in a PDF document."""

    text: str

    def to_json(self) -> dict:
        """Convert the MaterialDescriptionLine object to a JSON serializable dictionary."""
        return {"text": self.text}

    @classmethod
    def from_json(cls, data: dict) -> Self:
        """Converts a dictionary to an object."""
        return cls(text=data["text"])


@dataclass
class MaterialDescription(ExtractedFeature):
    """Class to represent a material description in a PDF document."""

    text: str
    lines: list[FeatureOnPage[MaterialDescriptionLine]]
    max_line_width: float | None = None

    @property
    def rects_with_pages(self) -> list[RectWithPage]:
        """Get the bounding rectangles of the material description."""
        all_pages = {line.page_number for line in self.lines}
        if not all_pages:
            return []
        return [
            RectWithPage(
                rect=pymupdf.Rect(
                    min(line.rect_with_page.rect.x0 for line in self.lines if line.page_number == page),
                    min(line.rect_with_page.rect.y0 for line in self.lines if line.page_number == page),
                    max(line.rect_with_page.rect.x1 for line in self.lines if line.page_number == page),
                    max(line.rect_with_page.rect.y1 for line in self.lines if line.page_number == page),
                ),
                page_number=page,
            )
            for page in sorted(list(all_pages))
        ]

    @property
    def pages(self) -> list[int]:
        return sorted(p_rect.page_number for p_rect in self.rects_with_pages)

    def rect_for_page(self, page_number: int) -> pymupdf.Rect | None:
        """Get the bounding rectangle for a specific page."""
        return next((p_rect.rect for p_rect in self.rects_with_pages if p_rect.page_number == page_number), None)

    @property
    def text_with_line_breaks(self) -> str:
        """Rejoin description lines with inferred line breaks, for display purposes only.

        Compares each line's width against `max_line_width` - the widest description line seen anywhere
        in this borehole - as a proxy for "how long a line can get before the layout wraps it": a line
        ending well short of that reference, or ending in sentence-final punctuation, is treated as an
        intentional line break rather than a layout wrap. A trailing period doesn't count as sentence-final
        punctuation when it's part of a known abbreviation (e.g. "ca.", "bzw.") - such lines only break via
        the length-based signal. Falls back to this description's own widest line when no borehole-wide
        reference was provided.
        """
        if not self.lines:
            return self.text
        reference_width = self.max_line_width or max((line.rect.width for line in self.lines), default=0)
        parts = [self.lines[0].feature.text]
        for prev_line, line in zip(self.lines, self.lines[1:], strict=False):
            gap_ratio = (reference_width - prev_line.rect.width) / reference_width if reference_width else 0.0
            prev_text = prev_line.feature.text.rstrip()
            ends_with_abbreviation = prev_text.endswith(_ABBREVIATIONS_ENDING_IN_PERIOD)
            ends_with_break_punct = prev_text.endswith((":", ";")) or (
                prev_text.endswith(".") and not ends_with_abbreviation
            )
            new_page = line.page_number != prev_line.page_number
            # TODO: 30% relative-to-longest-line threshold picked by eye, not tuned
            # yet; revisit once you've looked at a batch of extracted descriptions.
            is_break = new_page or ends_with_break_punct or gap_ratio > 0.3
            parts.append(("\n" if is_break else " ") + line.feature.text)
        return "".join(parts)

    def to_json(self) -> dict:
        """Convert the MaterialDescription object to a JSON serializable dictionary."""
        return {
            "text": self.text,
            "text_with_line_breaks": self.text_with_line_breaks,
            "lines": [line.to_json() for line in self.lines],
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_json(cls, data: dict) -> Self:
        """Converts a dictionary to an object."""
        return cls(
            text=data["text"],
            lines=[FeatureOnPage.from_json(line, MaterialDescriptionLine) for line in data["lines"]],
            is_correct=data.get("is_correct"),
        )


@dataclass
class TextBlock(RectWithPageMixin):
    """Class to represent a block of text in a PDF document.

    A TextBlock is a collection of Lines surrounded by Lines.
    It is used to represent a block of text in a PDF document.
    """

    lines: list[TextLine]
    is_terminated_by_line: bool = False

    def __post_init__(self):
        self.line_count = len(self.lines)
        self.text = " ".join([line.text for line in self.lines])
        if self.line_count:
            rect = pymupdf.Rect(
                min(line.rect.x0 for line in self.lines),
                min(line.rect.y0 for line in self.lines),
                max(line.rect.x1 for line in self.lines),
                max(line.rect.y1 for line in self.lines),
            )
        else:
            rect = pymupdf.Rect()

        # go through all the lines and check if they are on the same page
        page_number_set = set(line.page_number for line in self.lines)
        assert len(page_number_set) < 2, "TextBlock spans multiple pages"
        page_number = page_number_set.pop() if page_number_set else None
        self.rect_with_page = RectWithPage(rect, page_number)

    def concatenate(self, other: TextBlock) -> TextBlock:
        """Concatenate two text blocks.

        Args:
            other (TextBlock): The other text block.

        Returns:
            TextBlock: The concatenated text block.
        """
        new_lines = []
        new_lines.extend(self.lines)
        new_lines.extend(other.lines)
        return TextBlock(new_lines)

    def _is_legend(self) -> bool:
        """Check if the current block contains / is a legend.

        Note: deprecated method.

        Legends are characterized by having multiple lines of a single word (e.g. "sand", "kies", etc.). Furthermore
        these words are usually aligned in either the x or y direction.

        Returns:
            bool: Whether the block is or contains a legend.
        """
        y0_coordinates = []
        x0_coordinates = []
        number_horizontally_close = 0
        number_vertically_close = 0
        for line in self.lines:
            if len(line.text.split(" ")) == 1 and not any(
                char in line.text for char in [".", ",", ";", ":", "!", "?"]
            ):  # sometimes single words in text are delimited by a punctuation.
                if _is_close(line.rect.y0, y0_coordinates, 1):
                    number_horizontally_close += 1
                if _is_close(line.rect.x0, x0_coordinates, 1):
                    number_vertically_close += 1
                x0_coordinates.append(line.rect.x0)
                y0_coordinates.append(line.rect.y0)
        return number_horizontally_close > 1 or number_vertically_close > 2


def _is_close(a: float, b: list, tolerance: float) -> bool:
    return any(abs(a - c) < tolerance for c in b)
