"""Module for the AToBSidebar, which contains depth intervals defined like "0.2m - 1.3m"."""

from __future__ import annotations

from dataclasses import dataclass

import fitz

from stratigraphy.depthcolumn.depthcolumnentry import AToBDepthColumnEntry
from stratigraphy.lines.line import TextLine
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.interval import AToBInterval

from .interval_block_group import IntervalBlockGroup
from .sidebar import Sidebar


@dataclass
class AToBSidebar(Sidebar[AToBDepthColumnEntry]):
    """Represents a sidebar where the upper and lower depths of each layer are explicitly specified.

    Example::
        0 - 0.1m: xxx
        0.1 - 0.3m: yyy
        0.3 - 0.8m: zzz
        ...
    """

    entries: list[AToBDepthColumnEntry]

    def __repr__(self):
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return "AToBSidebar({})".format(", ".join([str(entry) for entry in self.entries]))

    def depth_intervals(self) -> list[AToBInterval]:
        return [AToBInterval(entry) for entry in self.entries]

    def break_on_mismatch(self) -> list[AToBSidebar]:
        """Breaks the sidebar into segments where the depths are not in an arithmetic progression.

        Returns:
            list[AToBSidebar]: A list of depth column segments.
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

        return [AToBSidebar(segment) for segment in segments]

    def is_valid(self) -> bool:
        """Checks if the sidebar is valid.

        An AToBSidebar is valid if it is strictly increasing and the depth intervals are significant.

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
            params (dict): A dictionary of relevant parameters.

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
            groups.append(IntervalBlockGroup(depth_intervals=[interval], blocks=matched_blocks))
        return groups
