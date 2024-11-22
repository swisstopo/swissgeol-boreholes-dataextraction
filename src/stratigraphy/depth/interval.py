"""This module contains dataclasses for (depth) intervals."""

from __future__ import annotations

import abc

import fitz

from stratigraphy.lines.line import TextLine
from stratigraphy.text.textblock import TextBlock

from .depthcolumnentry import DepthColumnEntry


class Interval(metaclass=abc.ABCMeta):
    """Abstract class for (depth) intervals."""

    def __init__(self, start: DepthColumnEntry | None, end: DepthColumnEntry | None):
        super().__init__()
        self.start = start
        self.end = end

    @property
    def start_value(self) -> float | None:
        """Get the start value of the interval."""
        if self.start:
            return self.start.value
        else:
            return None

    @property
    def end_value(self) -> float | None:
        """Get the end value of the interval."""
        if self.end:
            return self.end.value
        else:
            return None

    @property
    @abc.abstractmethod
    def line_anchor(self) -> fitz.Point:
        """Get the line anchor of the interval."""
        pass

    @property
    @abc.abstractmethod
    def background_rect(self) -> fitz.Rect | None:
        """Get the background rectangle of the interval."""
        pass

    def to_json(self):
        """Convert the interval to a JSON serializable format."""
        return {
            "start": self.start.to_json() if self.start else None,
            "end": self.end.to_json() if self.end else None,
        }


class AAboveBInterval(Interval):
    """Class for depth intervals where the upper depth is located above the lower depth on the page."""

    @property
    def line_anchor(self) -> fitz.Point | None:
        if self.start and self.end:
            return fitz.Point(self.start.rect.x1, (self.start.rect.y0 + self.end.rect.y1) / 2)
        elif self.start:
            return fitz.Point(self.start.rect.x1, self.start.rect.y1)
        elif self.end:
            return fitz.Point(self.end.rect.x1, self.end.rect.y0)

    @property
    def background_rect(self) -> fitz.Rect | None:
        if self.start and self.end:
            return fitz.Rect(self.start.rect.x0, self.start.rect.y1, self.start.rect.x1, self.end.rect.y0)

    def matching_blocks(
        self, all_blocks: list[TextBlock], block_index: int
    ) -> tuple[list[TextBlock], list[TextBlock], list[TextBlock]]:
        """Calculates pre, exact and post blocks for the boundary interval.

        Pre contains all the blocks that are supposed to come before the interval.
        Exact contains all the blocks that are supposed to be inside the interval.
        Post contains all the blocks that are supposed to come after the interval.

        Args:
            all_blocks (list[TextBlock]): All blocks available blocks.
            block_index (int): Index of the current block.

        Returns:
            tuple[list[TextBlock], list[TextBlock], list[TextBlock]]: Pre, exact and post blocks.
        """
        pre, exact, post = [], [], []

        while block_index < len(all_blocks) and (
            self.end is None or all_blocks[block_index].rect.y1 < self.end.rect.y1
        ):
            current_block = all_blocks[block_index]

            # only exact match when sufficient distance to previous and next blocks, to avoid a vertically shifted
            #  description "accidentally" being nicely contained in the depth interval.
            distances_above = [
                current_block.rect.y0 - other.rect.y1 for other in all_blocks if other.rect.y0 < current_block.rect.y0
            ]
            distance_above_ok_for_exact = len(distances_above) == 0 or min(distances_above) > 5

            exact_match_blocks = []
            exact_match_index = block_index
            if distance_above_ok_for_exact:
                continue_exact_match = True
                can_end_exact_match = True
                while continue_exact_match and exact_match_index < len(all_blocks):
                    exact_match_block = all_blocks[exact_match_index]
                    exact_match_rect = exact_match_block.rect
                    if (
                        self.start is None or exact_match_rect.y0 > (self.start.rect.y0 + self.start.rect.y1) / 2
                    ) and (self.end is None or exact_match_rect.y1 < (self.end.rect.y0 + self.end.rect.y1) / 2):
                        exact_match_blocks.append(exact_match_block)
                        exact_match_index += 1
                        distances_below = [other.rect.y0 - exact_match_block.rect.y1 for other in all_blocks]
                        distances_below = [distance for distance in distances_below if distance > 0]
                        can_end_exact_match = len(distances_below) == 0 or min(distances_below) > 5
                    else:
                        continue_exact_match = False

                if not can_end_exact_match:
                    exact_match_blocks = []

            if exact_match_blocks:
                exact.extend(exact_match_blocks)
                block_index = exact_match_index - 1
            elif exact:
                post.append(current_block)
            else:
                pre.append(current_block)

            block_index += 1

        return pre, exact, post


class AToBInterval(Interval):
    """Class for intervals that are defined in a single line like "1.00 - 2.30m"."""

    def __init__(self, start: DepthColumnEntry, end: DepthColumnEntry):
        super().__init__(start, end)

    @property
    def rect(self) -> fitz.Rect:
        """Get the rectangle surrounding the interval."""
        return fitz.Rect(self.start.rect).include_rect(self.end.rect)

    @property
    def line_anchor(self) -> fitz.Point | None:
        if self.end:
            return fitz.Point(self.end.rect.x1, (self.end.rect.y0 + self.end.rect.y1) / 2)

    @property
    def background_rect(self) -> fitz.Rect | None:
        return None

    def matching_blocks(
        self, all_lines: list[TextLine], line_index: int, next_interval: Interval | None
    ) -> list[TextBlock]:
        """Adds lines to a block until the next layer identifier is reached."""
        y1_threshold = None
        if next_interval:
            next_interval_start_rect = next_interval.start.rect
            y1_threshold = next_interval_start_rect.y0 + next_interval_start_rect.height / 2

        matched_lines = []

        for current_line in all_lines[line_index:]:
            if y1_threshold is None or current_line.rect.y1 < y1_threshold:
                matched_lines.append(current_line)
            else:
                break

        if matched_lines:
            return [TextBlock(matched_lines)]
        else:
            return []
