"""This module contains dataclasses for (depth) intervals."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import fitz

from stratigraphy.util.depthcolumnentry import DepthColumnEntry, LayerDepthColumnEntry
from stratigraphy.util.line import TextLine
from stratigraphy.util.textblock import TextBlock


class Interval(metaclass=abc.ABCMeta):
    """Abstract class for (depth) intervals."""

    def __init__(self, start: DepthColumnEntry | None, end: DepthColumnEntry | None):
        super().__init__()
        self.start = start
        self.end = end

    @property
    def start_value(self) -> float | None:
        if self.start:
            return self.start.value
        else:
            return None

    @property
    def end_value(self) -> float | None:
        if self.end:
            return self.end.value
        else:
            return None

    @property
    @abc.abstractmethod
    def line_anchor(self) -> fitz.Point:
        pass

    @property
    @abc.abstractmethod
    def background_rect(self) -> fitz.Rect | None:
        pass

    def to_json(self):
        return {
            "start": self.start.to_json() if self.start else None,
            "end": self.end.to_json() if self.end else None,
        }


@dataclass
class AnnotatedInterval:
    """Class for annotated intervals."""

    start: float
    end: float
    background_rect: fitz.Rect


class BoundaryInterval(Interval):
    """Class for boundary intervals.

    Boundary intervals are intervals that are defined by a start and an end point.
    """

    def __init__(self, start: DepthColumnEntry | None, end: DepthColumnEntry | None):
        super().__init__(start, end)

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

    def matching_blocks(self, all_blocks: list[TextBlock], block_index: int) -> tuple[list[TextBlock]]:
        """Calculates pre, exact and post blocks for the boundary interval.

        Pre contains all the blocks that are supposed to come before the interval.
        Exact contains all the blocks that are supposed to be inside the interval.
        Post contains all the blocks that are supposed to come after the interval.

        Args:
            all_blocks (list[TextBlock]): All blocks available blocks.
            block_index (int): Index of the current block.

        Returns:
            tuple[list[TextBlock]]: Pre, exact and post blocks.
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

            if len(exact_match_blocks):
                exact.extend(exact_match_blocks)
                block_index = exact_match_index - 1
            elif len(exact):
                post.append(current_block)
            else:
                pre.append(current_block)

            block_index += 1

        return pre, exact, post


class LayerInterval(Interval):
    """Class for layer intervals.

    A layer interval is an interval whose start and end-points are defined in a single entry.
    E.g. 1.00 - 2.30m.
    """

    def __init__(self, layer_depth_column_entry: LayerDepthColumnEntry):
        self.entry = layer_depth_column_entry
        super().__init__(layer_depth_column_entry.start, layer_depth_column_entry.end)

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

        if len(matched_lines):
            return [TextBlock(matched_lines)]
        else:
            return []
