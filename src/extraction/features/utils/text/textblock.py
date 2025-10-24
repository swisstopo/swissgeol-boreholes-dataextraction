"""This module contains the TextBlock class, which represents a block of text in a PDF document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import pymupdf

from extraction.features.utils.data_extractor import (
    ExtractedFeature,
    FeatureOnPage,
)
from extraction.features.utils.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin

from .textline import TextLine


@dataclass
class MaterialDescriptionLine(ExtractedFeature):
    """Class to represent a line of a material description in a PDF document."""

    text: str

    def to_json(self):
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

    def to_json(self):
        """Convert the MaterialDescription object to a JSON serializable dictionary."""
        return {"text": self.text, "lines": [line.to_json() for line in self.lines]}

    @classmethod
    def from_json(cls, data: dict) -> Self:
        """Converts a dictionary to an object."""
        return cls(
            text=data["text"], lines=[FeatureOnPage.from_json(line, MaterialDescriptionLine) for line in data["lines"]]
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
