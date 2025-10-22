"""Module for the AToBSidebar, which contains depth intervals defined like "0.2m - 1.3m"."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from extraction.features.stratigraphy.interval.interval import AToBInterval, IntervalBlockPair, IntervalZone
from extraction.features.stratigraphy.interval.partitions_and_sublayers import (
    number_of_subintervals,
    set_interval_hierarchy_flags,
)
from extraction.features.utils.text.textblock import TextBlock
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

    kind: ClassVar[str] = "a_to_b"

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

            # It seems reasonable that a single "parent" layer should not have more than 8 sublayers. This check
            # ensures that when e.g. a scale "1:100" is misinterpreted as a parent layer from 1m to 100m, the entire
            # borehole profile below is not incorrectly reduced to being sublayers of this incorrectly extracted layer.
            if sublayer_count > 8:
                depths_ok = False
            else:
                if index + sublayer_count + 1 >= len(self.entries):
                    depths_ok = True
                else:
                    next_interval = self.entries[index + sublayer_count + 1]
                    if sublayer_count == 0:
                        # no subintervals, the next interval must start deeper or at the same depth than the end of
                        # the current interval
                        depths_ok = current_interval.end.value <= next_interval.start.value
                    else:
                        # subintervals, the next interval should start exactly at the end of the current interval
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
            set_interval_hierarchy_flags(segment.entries)
            sidebar_list.append(AToBSidebar(segment.entries))

        return sidebar_list

    def is_valid(self) -> bool:
        """Checks if the sidebar is valid.

        An AToBSidebar is valid if its depth intervals are mostly increasing and are significant.
        This function only consider the effective intervals (i.e. not the parents or the sublayers).

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

        return sequence_matches_count / (len(filtered_entries) - 1) >= 0.5

    @staticmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines to AToBInterval zones.

        The score is 1.0 if the line is located within the interval zone, 0.0 otherwise.
        For AtoB sidebar, the zone begins and ends at the top of each rectangle bounds.

        Args:
            interval_zone (IntervalZone): The interval zone to score against.
            line (TextLine): The text line to score.

        Returns:
            float: The score for the given interval zone and text line.
        """
        start_top = interval_zone.start.y0 if interval_zone.start else None
        end_top = interval_zone.end.y0 if interval_zone.end else None
        line_mid = (line.rect.y0 + line.rect.y1) / 2
        if (start_top is None or line_mid > start_top) and (end_top is None or line_mid < end_top):
            return 1.0  # textline is inside the depth interval
        return 0.0

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        The interval zones are created from the AToBInterval entries, filtering out sublayers and invalid layers.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        filtered_entries = [entry for entry in self.entries if not entry.is_sublayer]
        filtered_entries = [entry for entry in filtered_entries if not (entry.start is None and entry.end.value == 0)]
        if not filtered_entries:
            return []
        zones = [
            IntervalZone(entry.rect, next_entry.rect, entry)
            for entry, next_entry in zip(filtered_entries, filtered_entries[1:], strict=False)
        ]
        return zones + [IntervalZone(filtered_entries[-1].rect, None, filtered_entries[-1])]  # last one is open-ended

    def post_processing(self, interval_lines_mapping: list[tuple[IntervalZone, list[TextLine]]]):
        """Post-process the matched interval zones and description lines into IntervalBlockPairs.

        The post-processing filters out layers that are marked as parents.

        Args:
            interval_lines_mapping (list[tuple[IntervalZone, list[TextLine]]]): The matched interval zones and
                description lines.

        Returns:
            list[IntervalBlockPair]: The processed interval block pairs.
        """
        valid_layers = [layer for layer in interval_lines_mapping if not layer[0].related_interval.is_parent]
        return [IntervalBlockPair(zone.related_interval, TextBlock(lines)) for zone, lines in valid_layers]
