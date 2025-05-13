"""Module for the AToBSidebar, which contains depth intervals defined like "0.2m - 1.3m"."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
from extraction.features.stratigraphy.interval.interval import AToBInterval, IntervalBlockGroup
from extraction.features.stratigraphy.interval.partitions_and_sublayers import (
    annotate_intervals,
    number_of_subintervals,
)
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.textline import TextLine

from .sidebar import Sidebar


@dataclass
class AToBSidebar(Sidebar[AToBInterval]):
    """Represents a sidebar where the upper and lower depths of each layer are explicitly specified.

    Example::
        0 - 0.1m: xxx
        0.1 - 0.3m: yyy
        0.3 - 0.8m: zzz
        ...
    """

    entries: list[AToBInterval]

    def __repr__(self):
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return "AToBSidebar({})".format(", ".join([str(entry) for entry in self.entries]))

    def break_on_mismatch(self) -> list[AToBSidebar]:
        """Breaks the sidebar into segments where the depths clearly don't belong to the same boreholes.

        This method tries to take into account cases where the depths are not always increasing, e.g. because of
        partitioned layers / sublayers.

        This method might become redundant after we implement a more reliable detection/separation of borehole profiles
        in the isssue https://github.com/swisstopo/swissgeol-boreholes-dataextraction/issues/195

        Returns:
            list[AToBSidebar]: A list of depth column segments.
        """
        segments = []
        segment_start = 0
        index = 0
        while index < len(self.entries):
            # We allow sublayers with depths lower than the end of the current entry, as long as the next layer
            # has a start depth that exactly matches the current end depth.
            current_interval = self.entries[index]
            sublayer_count = number_of_subintervals(current_interval, self.entries[index + 1 :])

            if sublayer_count > 8:
                depths_ok = False
            else:
                if index + sublayer_count + 1 >= len(self.entries):
                    depths_ok = True
                else:
                    next_interval = self.entries[index + sublayer_count + 1]
                    if sublayer_count == 0:
                        # no subintervals, the next interval should not start higher than the end of the current
                        # interval
                        depths_ok = current_interval.end.value <= next_interval.start.value
                    else:
                        # no subintervals, the next interval should start exactly at the end of the current interval
                        depths_ok = current_interval.end.value == next_interval.start.value

            if depths_ok:
                # all good, continue with the next interval
                index += sublayer_count + 1
            else:
                segments.append(self.entries[segment_start : index + 1])
                index += 1
                segment_start = index

        final_segment = self.entries[segment_start:]
        if final_segment:
            segments.append(final_segment)

        return [AToBSidebar(segment) for segment in segments]

    def process(self) -> list[AToBSidebar]:
        sidebar_list = []
        for segment in self.break_on_mismatch():
            annotate_intervals(segment.entries)
            sidebar_list.append(AToBSidebar(segment.entries))

        return sidebar_list

    def is_valid(self) -> bool:
        """Checks if the sidebar is valid.

        An AToBSidebar is valid if it is strictly increasing and the depth intervals are significant.

        Returns:
            bool: True if the depth column is valid, False otherwise.
        """
        filtered_entries = [entry for entry in self.entries if not entry.skip_interval]
        if len(filtered_entries) <= 2:
            return False

        # At least half of the "end" values must match the subsequent "start" value (e.g. 2-5m, 5-9m).
        sequence_matches_count = 0
        for index, entry in enumerate(filtered_entries):
            if index >= 1 and filtered_entries[index - 1].end.value == entry.start.value:
                sequence_matches_count += 1

        return sequence_matches_count / (len(filtered_entries) - 1) > 0.5

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: pymupdf.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (pymupdf.Rect): The bounding box of the material description.
            params (dict): A dictionary of relevant parameters.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.
        """
        groups = []
        line_index = 0

        filtered_entries = [entry for entry in self.entries if not entry.is_sublayer]

        for interval_index, interval in enumerate(filtered_entries):
            # don't allow a layer above depth 0
            if interval.start is None and interval.end.value == 0:
                continue

            next_interval = (
                filtered_entries[interval_index + 1] if interval_index + 1 < len(filtered_entries) else None
            )

            matched_blocks = interval.matching_blocks(description_lines, line_index, next_interval)
            line_index += sum([len(block.lines) for block in matched_blocks])

            if not interval.is_parent:
                groups.append(IntervalBlockGroup(depth_intervals=[interval], blocks=matched_blocks))
        return groups
