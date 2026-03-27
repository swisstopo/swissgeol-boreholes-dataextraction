"""Module for the ProtocolSidebar, which occurs in boreholes of the "Bohrprotokoll" type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from extraction.features.stratigraphy.interval.interval import IntervalBlockPair, IntervalZone
from extraction.features.stratigraphy.sidebar.classes.depth_column_entry_sidebar import DepthColumEntrySidebar
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine


@dataclass
class ProtocolSidebar(DepthColumEntrySidebar):
    """Represents a sidebar for boreholes of the "Bohrprotokoll" type.

    In contrast to ``AAboveBSidebar``, the vertical position of the depth labels is not
    required to be proportional to their numerical value. Matching to the material
    description is therefore based on the visual row structure only.
    """

    kind: ClassVar[str] = "protocol"

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        # intervalzone = [IntervalZone(
        #                 interval.start.rect if interval.start else None,
        #                 interval.end.rect if interval.end else None,
        #                 interval,
        #             )
        #             for interval in self.depth_intervals()]
        # print("IntervalZone:", intervalzone)
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
        return DepthColumEntrySidebar.default_score(interval_zone, line)

    def is_valid(self) -> bool:
        """Checks if the sidebar is valid.

        A ProtocolSidebar is valid if it contains at least two entries and if its
        depth values are strictly increasing.

        Returns:
            bool: True if the sidebar is valid, False otherwise.
        """
        return len(self.entries) >= 2 and self.is_strictly_increasing()

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
