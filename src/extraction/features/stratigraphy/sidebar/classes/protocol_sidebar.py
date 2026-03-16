"""Module for the ProtocolSidebar, which occurs in boreholes of the "Bohrprotokoll" type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.interval import AAboveBInterval, IntervalBlockPair, IntervalZone
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine


@dataclass
class ProtocolSidebar(Sidebar[DepthColumnEntry]):
    """Represents a sidebar for boreholes of the "Bohrprotokoll" type.

    In contrast to ``AAboveBSidebar``, the vertical position of the depth labels is not
    required to be proportional to their numerical value. Matching to the material
    description is therefore based on the visual row structure only.
    """

    entries: list[DepthColumnEntry]
    skipped_entries: list[DepthColumnEntry] = field(default_factory=list)

    kind: ClassVar[str] = "protocol"

    def __repr__(self):
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return "ProtocolSidebar({})".format(", ".join([str(entry) for entry in self.entries]))

    def strictly_contains(self, other: ProtocolSidebar) -> bool:
        """Check whether this sidebar strictly contains another one.

        Args:
            other (ProtocolSidebar): The other sidebar.

        Returns:
            bool: True if this sidebar strictly contains the other one.
        """
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_strictly_increasing(self) -> bool:
        """Check whether the depth values are strictly increasing.

        Returns:
            bool: True if the depth values are strictly increasing.
        """
        return all(self.entries[i].value < self.entries[i + 1].value for i in range(len(self.entries) - 1))

    def depth_intervals(self) -> list[AAboveBInterval]:
        """Creates a list of depth intervals from the depth column entries.

        The first depth interval has an open start value (i.e. None) if the first
        depth entry is not 0.0.

        Returns:
            list[AAboveBInterval]: A list of depth intervals.
        """
        depth_intervals = []
        if self.entries[0].value != 0.0:
            depth_intervals.append(AAboveBInterval(None, self.entries[0]))
        for i in range(len(self.entries) - 1):
            depth_intervals.append(AAboveBInterval(self.entries[i], self.entries[i + 1]))
        return depth_intervals

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        return [
            IntervalZone(
                interval.start.rect if interval.start else None,
                interval.end.rect if interval.end else None,
                interval,
            )
            for interval in self.depth_intervals()
        ]

    def trim_trailing_duplicate_depths(self) -> ProtocolSidebar:
        """Drop trailing duplicates of the final depth value.

        This handles common protocol-table cases where the final depth (e.g. Endtiefe)
        is repeated at the bottom. Only trailing duplicates of the last value are removed.
        """
        entries = list(self.entries)

        if len(entries) < 2:
            return self

        last_val = entries[-1].value

        while len(entries) > 1 and entries[-2].value == last_val:
            entries.pop()

        if len(entries) == len(self.entries):
            return self

        return ProtocolSidebar(entries, self.skipped_entries)

    @staticmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines.

        The score is 1.0 if the line is located within the interval zone, 0.0 otherwise.
        For ProtocolSidebar, the zone begins and ends at the top of each rectangle
        bounds, similarly to ``AToBSidebar``.

        Args:
            interval_zone (IntervalZone): The interval zone to score against.
            line (TextLine): The text line to score.

        Returns:
            float: The score for the given interval zone and text line.
        """
        return Sidebar.default_score(interval_zone, line)

    def is_valid(self) -> bool:
        """Checks if the sidebar is valid.

        A ProtocolSidebar is valid if it contains at least two entries and if its
        depth values are strictly increasing.

        Returns:
            bool: True if the sidebar is valid, False otherwise.
        """
        return len(self.entries) >= 2 and self.is_strictly_increasing()

    def break_on_double_descending(self) -> list[ProtocolSidebar]:
        """Break the sidebar into segments when two consecutive descending entries occur.

        This is a lightweight safeguard against accidentally merged clusters from
        different boreholes or unrelated table sections.

        Returns:
            list[ProtocolSidebar]: A list of sidebar segments.
        """
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
                segments.append(self.entries[segment_start:index])
                segment_start = index

        final_segment = self.entries[segment_start:]
        if final_segment:
            segments.append(final_segment)

        return [ProtocolSidebar(segment, self.skipped_entries) for segment in segments]

    def process(self) -> list[ProtocolSidebar]:
        """Post-process the sidebar into valid segments.

        Returns:
            list[ProtocolSidebar]: A list of processed sidebars.
        """
        processed_sidebars = []
        for sidebar in self.break_on_double_descending():
            trimmed_sidebar = sidebar.trim_trailing_duplicate_depths()
            if trimmed_sidebar.is_valid():
                processed_sidebars.append(trimmed_sidebar)
        return processed_sidebars

    def post_processing(
        self, interval_lines_mapping: list[tuple[IntervalZone, list[TextLine]]]
    ) -> list[IntervalBlockPair]:
        """Post-process the matched interval zones and description lines.

        Args:
            interval_lines_mapping (list[tuple[IntervalZone, list[TextLine]]]): The matched
                interval zones and description lines.

        Returns:
            list[IntervalBlockPair]: The processed interval block pairs.
        """
        return [IntervalBlockPair(zone.related_interval, TextBlock(lines)) for zone, lines in interval_lines_mapping]
