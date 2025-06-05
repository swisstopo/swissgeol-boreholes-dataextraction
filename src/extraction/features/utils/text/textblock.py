"""This module contains the TextBlock class, which represents a block of text in a PDF document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pymupdf
from extraction.features.utils.data_extractor import (
    ExtractedFeature,
    FeatureOnPage,
)

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
        if page_number_set:
            self.page_number = page_number_set.pop()
        else:
            self.page_number = None

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

    # LGD-288: sometimes indentation is the only significant signal for deciding where we need to split the material
    # descriptions of adjacent layers.
    def split_based_on_indentation(self) -> list[TextBlock]:
        """Split the text block based on indentation.

        Returns:
            list[TextBlock]: The split text blocks.
        """
        if len(self.lines) == 0:
            return []

        line_starts = [line.rect.x0 for line in self.lines]
        min_line_start = min(line_starts)
        max_line_width = max([line.rect.width for line in self.lines])

        first_line_start = self.lines[0].rect.x0
        indentation_low = min_line_start + 0.03 * max_line_width
        indentation_high = min_line_start + 0.2 * max_line_width

        # don't do anything if the first line already indented (e.g. centered text)
        if first_line_start > indentation_low:
            return [self]
        # don't do anything if we don't have any lines at a reasonable indentation
        # (2%-20% of max width from leftmost edge)
        if all(line.rect.x0 < indentation_low or line.rect.x0 > indentation_high for line in self.lines):
            return [self]

        # split based on indentation
        blocks = []
        current_block_lines = []
        for line in self.lines:
            if line.rect.x0 < indentation_low:
                # start new block
                if current_block_lines:
                    blocks.append(TextBlock(current_block_lines))
                current_block_lines = [line]
            else:
                # continue block
                current_block_lines.append(line)

        if current_block_lines:
            blocks.append(TextBlock(current_block_lines))

        if self.is_terminated_by_line:  # if the block was terminated by a line, then the last block should be as well
            blocks[-1].is_terminated_by_line = True
        return blocks

    def _is_legend(self) -> bool:
        """Check if the current block contains / is a legend.

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


def block_distance(block1: TextBlock, block2: TextBlock) -> float:
    """Calculate the distance between two text blocks.

    The distance is calculated as the difference between the y-coordinates of the bottom of the first block
    and the top of the second block.

    If a block is terminated by a line, the distance to the next block is set to infinity.
    This ensures that the block is not merged with the next block.

    Args:
        block1 (TextBlock): The first text block.
        block2 (TextBlock): The second text block.

    Returns:
        float: The distance between the two text blocks.
    """
    if block1.is_terminated_by_line:
        return np.inf
    else:
        return block2.rect.y0 - block1.rect.y1
