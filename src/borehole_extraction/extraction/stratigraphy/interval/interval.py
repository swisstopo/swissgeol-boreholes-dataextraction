"""This module contains dataclasses for (depth) intervals."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
from borehole_extraction.extraction.stratigraphy.base_sidebar_entry.sidebar_entry import DepthColumnEntry
from borehole_extraction.extraction.util_extraction.text.textblock import TextBlock
from borehole_extraction.extraction.util_extraction.text.textline import TextLine


class DepthInterval:
    """class for (depth) intervals."""

    def __init__(self, start: DepthColumnEntry | None, end: DepthColumnEntry | None):
        super().__init__()
        self.start = start
        self.end = end

    def __repr__(self):
        return f"({self.start}, {self.end})"

    @property
    def line_anchor(self) -> pymupdf.Point | None:
        if self.start and self.end:
            return pymupdf.Point(
                max(self.start.rect.x1, self.end.rect.x1), (self.start.rect.y0 + self.end.rect.y1) / 2
            )
        elif self.start:
            return pymupdf.Point(self.start.rect.x1, self.start.rect.y1)
        elif self.end:
            return pymupdf.Point(self.end.rect.x1, self.end.rect.y0)

    @property
    def background_rect(self) -> pymupdf.Rect | None:
        if self.start and self.end and self.start.rect.y1 < self.end.rect.y0:
            return pymupdf.Rect(
                self.start.rect.x0, self.start.rect.y1, max(self.start.rect.x1, self.end.rect.x1), self.end.rect.y0
            )

    def to_json(self):
        """Convert the LayerDepths object to a JSON serializable format."""
        return {"start": self.start.to_json() if self.start else None, "end": self.end.to_json() if self.end else None}

    @classmethod
    def from_json(cls, data: dict) -> DepthInterval:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths.

        Returns:
            Interval: the corresponding LayerDepths object.
        """
        return cls(
            start=DepthColumnEntry.from_json(data["start"]) if data["start"] else None,
            end=DepthColumnEntry.from_json(data["end"]) if data["end"] else None,
        )


@dataclass
class IntervalBlockGroup:
    """Helper class to represent a group of depth intervals and an associated group of text blocks.

    The class is used to simplify the code for obtaining an appropriate one-to-one correspondence between depth
    intervals and material descriptions.
    """

    depth_intervals: list[DepthInterval]
    blocks: list[TextBlock]


@dataclass
class IntervalBlockPair:
    """Represent the data for a single layer in the borehole profile.

    This consist of a material description (represented as a text block) and a depth interval (if available).
    """

    depth_interval: DepthInterval | None
    block: TextBlock


class AAboveBInterval(DepthInterval):
    """Class for depth intervals where the upper depth is located above the lower depth on the page."""

    def matching_blocks(
        self, all_blocks: list[TextBlock], block_index: int, min_block_clearance: int
    ) -> tuple[list[TextBlock], list[TextBlock], list[TextBlock]]:
        """Calculates pre, exact and post blocks for the boundary interval.

        Pre contains all the blocks that are supposed to come before the interval.
        Exact contains all the blocks that are supposed to be inside the interval.
        Post contains all the blocks that are supposed to come after the interval.

        Args:
            all_blocks (list[TextBlock]): All blocks available blocks.
            block_index (int): Index of the current block.
            min_block_clearance (int): The required space above and below a block to a have an exact match.

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
            distance_above_ok_for_exact = len(distances_above) == 0 or min(distances_above) > min_block_clearance

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
                        can_end_exact_match = len(distances_below) == 0 or min(distances_below) > min_block_clearance
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


class AToBInterval(DepthInterval):
    """Class for intervals that are defined in a single line like "1.00 - 2.30m"."""

    def __init__(self, start: DepthColumnEntry, end: DepthColumnEntry):
        super().__init__(start, end)

    def __repr__(self):
        return f"({self.start}, {self.end})"

    @property
    def rect(self) -> pymupdf.Rect:
        """Get the rectangle surrounding the interval."""
        return pymupdf.Rect(self.start.rect).include_rect(self.end.rect)

    def matching_blocks(
        self, all_lines: list[TextLine], line_index: int, next_interval: DepthInterval | None
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
