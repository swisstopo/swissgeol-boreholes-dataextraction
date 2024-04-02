"""This module contains the TextBlock class, which represents a block of text in a PDF document."""

from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np

from stratigraphy.util.line import TextLine


@dataclass
class TextBlock:
    """Class to represent a block of text in a PDF document."""

    lines: list[TextLine]
    is_terminated_by_line: bool = False

    def __post_init__(self):
        self.line_count = len(self.lines)
        self.text = " ".join([line.text for line in self.lines])
        if self.line_count:
            self.rect = fitz.Rect(
                min(line.rect.x0 for line in self.lines),
                min(line.rect.y0 for line in self.lines),
                max(line.rect.x1 for line in self.lines),
                max(line.rect.y1 for line in self.lines),
            )
        else:
            self.rect = fitz.Rect()

    def concatenate(self, other: TextBlock):
        new_lines = []
        new_lines.extend(self.lines)
        new_lines.extend(other.lines)
        return TextBlock(new_lines)

    # LGD-288: sometimes indentation is the only significant signal for deciding where we need to split the material
    # descriptions of adjacent layers.
    def split_based_on_indentation(self) -> list[TextBlock]:
        if len(self.lines) == 0:
            return []

        line_starts = [line.rect.x0 for line in self.lines]
        min_line_start = min(line_starts)
        max_line_width = max([line.rect.width for line in self.lines])

        first_line_start = self.lines[0].rect.x0
        indentation_low = min_line_start + 0.02 * max_line_width
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
                if len(current_block_lines):
                    blocks.append(TextBlock(current_block_lines))
                current_block_lines = [line]
            else:
                # continue block
                current_block_lines.append(line)

        if len(current_block_lines):
            blocks.append(TextBlock(current_block_lines))

        if self.is_terminated_by_line:  # if the block was terminated by a line, then the last block should be as well
            blocks[-1].is_terminated_by_line = True
        return blocks

    def _is_legend(self) -> bool:
        y0_coordinates = []
        x0_coordinates = []
        number_horizontally_close = 0
        number_vertically_close = 0
        for line in self.lines:
            if line._is_legend_word():
                print(line.text)
                if _is_close(line.rect.y0, y0_coordinates, 1):
                    number_horizontally_close += 1
                if _is_close(line.rect.x0, x0_coordinates, 1):
                    number_vertically_close += 1
                x0_coordinates.append(line.rect.x0)
                y0_coordinates.append(line.rect.y0)
        return number_horizontally_close > 1 or number_vertically_close > 1

    def to_json(self):
        return {
            "text": self.text,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "lines": [line.to_json() for line in self.lines],
        }


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
