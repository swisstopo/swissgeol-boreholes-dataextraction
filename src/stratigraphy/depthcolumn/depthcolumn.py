"""This module contains the DepthColumn class, which is used to represent a depth column in a pdf page."""

from __future__ import annotations

import abc

import fitz
import numpy as np
from stratigraphy.depthcolumn.depthcolumnentry import DepthColumnEntry, LayerDepthColumnEntry
from stratigraphy.layer.layer import IntervalBlockGroup
from stratigraphy.layer.layer_identifier_column import LayerIdentifierColumn
from stratigraphy.lines.line import TextLine, TextWord
from stratigraphy.text.find_description import get_description_blocks
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.interval import BoundaryInterval, Interval, LayerInterval


class DepthColumn(metaclass=abc.ABCMeta):
    """Abstract DepthColumn class."""

    @abc.abstractmethod
    def __init__(self):  # noqa: D107
        pass

    @abc.abstractmethod
    def depth_intervals(self) -> list[Interval]:
        """Get the depth intervals of the depth column."""
        pass

    @abc.abstractmethod
    def rects(self) -> list[fitz.Rect]:
        """Get the rectangles of the depth column entries."""
        pass

    """Used for scoring how well a depth column corresponds to a material description bbox."""

    def rect(self) -> fitz.Rect:
        """Get the bounding box of the depth column entries."""
        x0 = min([rect.x0 for rect in self.rects()])
        x1 = max([rect.x1 for rect in self.rects()])
        y0 = min([rect.y0 for rect in self.rects()])
        y1 = max([rect.y1 for rect in self.rects()])
        return fitz.Rect(x0, y0, x1, y1)

    @property
    def max_x0(self) -> float:
        """Get the maximum x0 value of the depth column entries."""
        return max([rect.x0 for rect in self.rects()])

    @property
    def min_x1(self) -> float:
        """Get the minimum x1 value of the depth column entries."""
        return min([rect.x1 for rect in self.rects()])

    @abc.abstractmethod
    def noise_count(self, all_words: list[TextWord]) -> int:
        """Count the number of words that intersect with the depth column entries.

        Args:
            all_words (list[TextWord]): A list of all text lines on the page.

        Returns:
            int: The number of words that intersect with the depth column entries but are not part of it.
        """
        pass

    @abc.abstractmethod
    def identify_groups(
        self, description_lines: list[TextLine], geometric_lines: list[Line], material_description_rect: fitz.Rect
    ) -> list[dict]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.

        Returns:
            list[dict]: A list of groups, where each group is a dictionary
                        with the keys "depth_intervals" and "blocks".
        """
        pass

    @abc.abstractmethod
    def to_json(self):
        """Converts the object to a dictionary."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_json(cls, json_depth_column: dict) -> DepthColumn:
        """Converts a dictionary to an object."""
        pass


class DepthColumnFactory:
    """Factory class for creating DepthColumn objects."""

    @staticmethod
    def create(data: dict) -> DepthColumn:
        """Creates a DepthColumn object from a dictionary.

        Args:
            data (dict): A dictionary representing the depth column.

        Returns:
            DepthColumn: The depth column object.
        """
        column_type = data.get("type")
        if column_type == "BoundaryDepthColumn":
            return BoundaryDepthColumn.from_json(data)
        elif column_type == "LayerDepthColumn":
            return LayerDepthColumn.from_json(data)
        elif column_type == "LayerIdentifierColumn":
            return LayerIdentifierColumn.from_json(data)
        else:
            raise ValueError(f"Unknown depth column type: {column_type}")


class LayerDepthColumn(DepthColumn):
    """Represents a depth column where the upper and lower depths of each layer are explicitly specified.

    Example::
        0 - 0.1m: xxx
        0.1 - 0.3m: yyy
        0.3 - 0.8m: zzz
        ...
    """

    entries: list[LayerDepthColumnEntry]

    def __init__(self, entries=None):
        super().__init__()

        if entries is not None:
            self.entries = entries
        else:
            self.entries = []

    def __repr__(self):
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return "LayerDepthColumn({})".format(", ".join([str(entry) for entry in self.entries]))

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        rect = self.rect()
        return {
            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
            "entries": [entry.to_json() for entry in self.entries],
            "type": "LayerDepthColumn",
        }

    @classmethod
    def from_json(cls, json_depth_column: dict) -> LayerDepthColumn:
        """Converts a dictionary to an object.

        Args:
            json_depth_column (dict): A dictionary representing the depth column.

        Returns:
            LayerDepthColumn: The depth column object.
        """
        entries_data = json_depth_column.get("entries", [])
        entries = [LayerDepthColumnEntry.from_json(entry) for entry in entries_data]
        return LayerDepthColumn(entries)

    def add_entry(self, entry: LayerDepthColumnEntry) -> LayerDepthColumn:
        """Adds a depth column entry to the depth column.

        Args:
            entry (LayerDepthColumnEntry): The depth column entry to add.

        Returns:
            LayerDepthColumn: The depth column with the new entry.
        """
        self.entries.append(entry)
        return self

    def depth_intervals(self) -> list[Interval]:
        return [LayerInterval(entry) for entry in self.entries]

    def rects(self) -> list[fitz.Rect]:
        return [entry.rect for entry in self.entries]

    def noise_count(self, all_words: list[TextWord]) -> int:
        # currently, we don't count noise for layer columns
        return 0

    def break_on_mismatch(self) -> list[LayerDepthColumn]:
        """Breaks the depth column into segments where the depth intervals are not in an arithmetic progression.

        Returns:
            list[LayerDepthColumn]: A list of depth column segments.
        """
        segments = []
        segment_start = 0
        for index, current_entry in enumerate(self.entries):
            if index >= 1 and current_entry.start.value < self.entries[index - 1].end.value:
                # (_, big) || (small, _)
                segments.append(self.entries[segment_start:index])
                segment_start = index

        final_segment = self.entries[segment_start:]
        if final_segment:
            segments.append(final_segment)

        return [LayerDepthColumn(segment) for segment in segments]

    def is_valid(self) -> bool:
        """Checks if the depth column is valid.

        A depth column is valid if it is strictly increasing and the depth intervals are significant.

        Returns:
            bool: True if the depth column is valid, False otherwise.
        """
        if len(self.entries) <= 2:
            return False

        # At least half of the "end" values must match the subsequent "start" value (e.g. 2-5m, 5-9m).
        sequence_matches_count = 0
        for index, entry in enumerate(self.entries):
            if index >= 1 and self.entries[index - 1].end.value == entry.start.value:
                sequence_matches_count += 1

        return sequence_matches_count / (len(self.entries) - 1) > 0.5

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: fitz.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.
            params (dict): A dictionary of parameters used for line detection.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.
        """
        depth_intervals = self.depth_intervals()

        groups = []
        line_index = 0

        for interval_index, interval in enumerate(depth_intervals):
            # don't allow a layer above depth 0
            if interval.start is None and interval.end.value == 0:
                continue

            next_interval = depth_intervals[interval_index + 1] if interval_index + 1 < len(depth_intervals) else None

            matched_blocks = interval.matching_blocks(description_lines, line_index, next_interval)
            line_index += sum([len(block.lines) for block in matched_blocks])
            groups.append(
                # TODO: This seems to be the only case where a list is passed and most of the time it is a list of one
                # element. Seem to need the function: transform_groups().
                IntervalBlockGroup(depth_interval=[interval], block=matched_blocks)
            )
        return groups


class BoundaryDepthColumn(DepthColumn):
    """Represents a depth column.

    The depths of the boundaries between layers are labels, at a vertical position on
    the page that is proportional to the depth.

    Example:
        0m

        0.2m


        0.5m
        ...
    """

    entries: list[DepthColumnEntry]

    def __init__(self, entries: list = None):
        """Initializes a BoundaryDepthColumn object.

        Args:
            entries (list, optional): Depth Column Entries for the depth column. Defaults to None.
        """
        super().__init__()

        if entries is not None:
            self.entries = entries
        else:
            self.entries = []

    def rects(self) -> list[fitz.Rect]:
        return [entry.rect for entry in self.entries]

    def __repr__(self):
        return "DepthColumn({})".format(", ".join([str(entry) for entry in self.entries]))

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        rect = self.rect()
        return {
            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
            "entries": [entry.to_json() for entry in self.entries],
            "type": "BoundaryDepthColumn",
        }

    @classmethod
    def from_json(cls, json_depth_column: dict) -> BoundaryDepthColumn:
        """Converts a dictionary to an object.

        Args:
            json_depth_column (dict): A dictionary representing the depth column.

        Returns:
            BoundaryDepthColumn: The depth column object.
        """
        entries_data = json_depth_column.get("entries", [])
        entries = [DepthColumnEntry.from_json(entry) for entry in entries_data]
        return BoundaryDepthColumn(entries)

    def add_entry(self, entry: DepthColumnEntry) -> BoundaryDepthColumn:
        """Adds a depth column entry to the depth column.

        Args:
            entry (DepthColumnEntry): The depth column entry to add.

        Returns:
            BoundaryDepthColumn: The depth column with the new entry.
        """
        self.entries.append(entry)
        return self

    """
    Check if the middle of the new rect is between the outer horizontal boundaries of the column, and if there is an
    intersection with the minimal horizontal boundaries of the column.
    """

    def can_be_appended(self, rect: fitz.Rect) -> bool:
        """Checks if a new depth column entry can be appended to the current depth column.

        The checks are:
        - The width of the new rectangle is greater than the width of the current depth column. Or;
        - The middle of the new rectangle is within the horizontal boundaries of the current depth column.
        - The new rectangle intersects with the minimal horizontal boundaries of the current depth column.


        Args:
            rect (fitz.Rect): Rect of the depth column entry to be appended.

        Returns:
            bool: True if the new depth column entry can be appended, False otherwise.
        """
        new_middle = (rect.x0 + rect.x1) / 2
        if (self.rect().width < rect.width or self.rect().x0 < new_middle < self.rect().x1) and (
            rect.x0 <= self.min_x1 and self.max_x0 <= rect.x1
        ):
            return True
        return False

    def valid_initial_segment(self, rect: fitz.Rect) -> BoundaryDepthColumn:
        for i in range(len(self.entries) - 1):
            initial_segment = BoundaryDepthColumn(self.entries[: -i - 1])
            if initial_segment.can_be_appended(rect):
                return initial_segment
        return BoundaryDepthColumn()

    def strictly_contains(self, other: BoundaryDepthColumn) -> bool:
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_strictly_increasing(self) -> bool:
        return all(i.value < j.value for i, j in zip(self.entries, self.entries[1:], strict=False))

    def depth_intervals(self) -> list[BoundaryInterval]:
        """Creates a list of depth intervals from the depth column entries.

        The first depth interval has an open start value (i.e. None).

        Returns:
            list[BoundaryInterval]: A list of depth intervals.
        """
        depth_intervals = [BoundaryInterval(None, self.entries[0])]
        for i in range(len(self.entries) - 1):
            depth_intervals.append(BoundaryInterval(self.entries[i], self.entries[i + 1]))
        depth_intervals.append(
            BoundaryInterval(self.entries[len(self.entries) - 1], None)
        )  # even though no open ended intervals are allowed, they are still useful for matching,
        # especially for documents where the material description rectangle is too tall
        # (and includes additional lines below the actual material descriptions).
        return depth_intervals

    def significant_arithmetic_progression(self) -> bool:
        # to allow for OCR errors or gaps in the progression, we only require a segment of length 6 that is an
        # arithmetic progression
        segment_length = 6
        if len(self.entries) < segment_length:
            return self.is_arithmetic_progression()
        else:
            for i in range(len(self.entries) - segment_length + 1):
                if BoundaryDepthColumn(self.entries[i : i + segment_length]).is_arithmetic_progression():
                    return True
            return False

    def is_arithmetic_progression(self) -> bool:
        if len(self.entries) <= 2:
            return True

        progression = np.array(range(len(self.entries)))
        entries = np.array([entry.value for entry in self.entries])

        # Avoid warnings in the np.corrcoef call, as the correlation coef is undefined if the standard deviation is 0.
        if np.std(entries) == 0:
            return False

        scale_pearson_correlation_coef = np.corrcoef(entries, progression)[0, 1].item()
        return abs(scale_pearson_correlation_coef) >= 0.9999

    def noise_count(self, all_words: list[TextWord]) -> int:
        """Counts the number of words that intersect with the depth column entries.

        Returns the number of words that intersect with the depth column entries, but are not part of the depth column.

        Args:
            all_words (list[TextWord]): A list of all text lines on the page.

        Returns:
            int: The number of words that intersect with the depth column entries but are not part of it.
        """

        def significant_intersection(other_rect):
            intersection = fitz.Rect(other_rect).intersect(self.rect())
            return intersection.is_valid and intersection.width > 0.25 * self.rect().width

        return len([word for word in all_words if significant_intersection(word.rect)]) - len(self.entries)

    def pearson_correlation_coef(self) -> float:
        # We look at the lower y coordinate, because most often the baseline of the depth value text is aligned with
        # the line of the corresponding layer boundary.
        positions = np.array([entry.rect.y1 for entry in self.entries])
        entries = np.array([entry.value for entry in self.entries])

        # Avoid warnings in the np.corrcoef call, as the correlation coef is undefined if the standard deviation is 0.
        if np.std(entries) == 0 or np.std(positions) == 0:
            return 0

        return np.corrcoef(positions, entries)[0, 1].item()

    def remove_entry_by_correlation_gradient(self) -> BoundaryDepthColumn | None:
        if len(self.entries) < 3:
            return None

        new_columns = [
            BoundaryDepthColumn([entry for index, entry in enumerate(self.entries) if index != remove_index])
            for remove_index in range(len(self.entries))
        ]
        return max(new_columns, key=lambda column: column.pearson_correlation_coef())

    def break_on_double_descending(self) -> list[BoundaryDepthColumn]:
        segments = []
        segment_start = 0
        for index, current_entry in enumerate(self.entries):
            if (
                index >= 2
                and index + 1 < len(self.entries)
                and current_entry.value < self.entries[index - 2].value
                and current_entry.value < self.entries[index - 1].value
                and self.entries[index + 1].value < self.entries[index - 2].value
                and self.entries[index + 1].value < self.entries[index - 1].value
            ):
                # big big || small small
                segments.append(self.entries[segment_start:index])
                segment_start = index

        final_segment = self.entries[segment_start:]
        if final_segment:
            segments.append(final_segment)

        return [BoundaryDepthColumn(segment) for segment in segments]

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: fitz.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Note: includes a heuristic of whether there should be a group corresponding to a final depth interval
        starting from the last depth entry without any end value.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.
            params (dict): A dictionary of parameters used for line detection.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.

        Example:
            [
                {
                    "depth_intervals": [BoundaryInterval(None, 0.1), BoundaryInterval(0.1, 0.3), ...],
                    "blocks": [DescriptionBlock(...), DescriptionBlock(...), ...]
                },
                {
                    "depth_intervals": [BoundaryInterval(0.3, 0.7)],
                    "blocks": [DescriptionBlock(...), DescriptionBlock(...), ...]
                },
                ...
            ]
        """
        depth_intervals = self.depth_intervals()

        groups = []

        current_intervals = []
        current_blocks = []
        all_blocks = get_description_blocks(
            description_lines,
            geometric_lines,
            material_description_rect,
            params["block_line_ratio"],
            left_line_length_threshold=params["left_line_length_threshold"],
            target_layer_count=len(depth_intervals),
        )

        block_index = 0

        for interval in depth_intervals:
            # don't allow a layer above depth 0
            if interval.start is None and interval.end.value == 0:
                continue

            pre, exact, post = interval.matching_blocks(all_blocks, block_index)
            block_index += len(pre) + len(exact) + len(post)

            current_blocks.extend(pre)
            if len(exact):
                if len(current_intervals) > 0 or len(current_blocks) > 0:
                    groups.append(IntervalBlockGroup(depth_interval=current_intervals, block=current_blocks))
                groups.append(IntervalBlockGroup(depth_interval=[interval], block=exact))
                current_blocks = post
                current_intervals = []
            else:
                # The final open-ended interval should not be added, since borehole profiles do typically not come
                # with open-ended intervals.
                if interval.end is not None:
                    current_intervals.append(interval)

        if len(current_intervals) > 0 or len(current_blocks) > 0:
            groups.append(IntervalBlockGroup(depth_interval=current_intervals, block=current_blocks))

        return groups
