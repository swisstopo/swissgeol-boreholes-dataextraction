"""This module contains the TextBlock class, which represents a block of text in a PDF document."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
from pydantic import model_serializer

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.data_extractor import (
    ExtractedFeature,
    FeatureOnPage,
)


# TODO: the correctness of an individual MaterialDescriptionLine is not evaluated. Therefore, the class
# should ideally not inherit from ExtractedFeature. However, this would then not allow us to define a
# FeatureOnPage[MaterialDescriptionLine]. In future, we should de-couple ExtractedFeature and
# FeatureOnPage, and afterwards remove the inheritance and the modified serialization logic here.
class MaterialDescriptionLine(ExtractedFeature):
    """Class to represent a line of a material description in a PDF document."""

    text: str

    @model_serializer(mode="wrap")
    def serialize_without_is_correct(self, handler) -> dict:
        serialized_dict = handler(self)
        serialized_dict.pop("is_correct")
        return serialized_dict


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
                    min(line.rect.x0 for line in self.lines if line.page_number == page),
                    min(line.rect.y0 for line in self.lines if line.page_number == page),
                    max(line.rect.x1 for line in self.lines if line.page_number == page),
                    max(line.rect.y1 for line in self.lines if line.page_number == page),
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

    def insert_line_breaks(self, max_line_width: float | None) -> MaterialDescription:
        """Rejoin description lines with inferred line breaks, for display purposes only.

        Compares each line's width against `max_line_width` - the widest description line seen anywhere
        in this borehole - as a proxy for "how long a line can get before the layout wraps it": a line
        ending well short of that reference, or ending in sentence-final punctuation, is treated as an
        intentional line break rather than a layout wrap. A trailing period doesn't count as sentence-final
        punctuation when it's part of a known abbreviation (e.g. "ca.", "bzw.") - such lines only break via
        the length-based signal. Falls back to this description's own widest line when no borehole-wide
        reference was provided. A vertical gap to the next line that's noticeably larger than the line's
        own height (e.g. a blank line in the layout) is also treated as a break, even if none of the other
        criteria apply.
        """
        if not self.lines:
            return self
        reference_width = max_line_width or max((line.rect.width for line in self.lines), default=0)

        for prev_line, line in zip(self.lines, self.lines[1:], strict=False):
            gap_ratio = (reference_width - prev_line.rect.width) / reference_width if reference_width else 0.0
            new_page = line.page_number != prev_line.page_number
            vertical_gap = line.rect.y0 - prev_line.rect.y1
            # TODO: 30% relative-to-longest-line and 0.5x-line-height thresholds picked by eye, not
            # tuned, works well enough so far
            has_large_vertical_gap = not new_page and vertical_gap > 0.5 * prev_line.rect.height
            is_break = new_page or gap_ratio > 0.3 or has_large_vertical_gap
            if is_break:
                prev_line.feature.text += "\n"
            else:
                prev_line.feature.text += " "

        self.text = "".join([line.feature.text for line in self.lines])
        return self


@dataclass
class TextBlock:
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
            self.rect = pymupdf.Rect(
                min(line.rect.x0 for line in self.lines),
                min(line.rect.y0 for line in self.lines),
                max(line.rect.x1 for line in self.lines),
                max(line.rect.y1 for line in self.lines),
            )
        else:
            self.rect = pymupdf.Rect()

        # go through all the lines and check if they are on the same page
        page_number_set = set(line.page_number for line in self.lines)
        assert len(page_number_set) < 2, "TextBlock spans multiple pages"
        self.page = page_number_set.pop() if page_number_set else None

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
