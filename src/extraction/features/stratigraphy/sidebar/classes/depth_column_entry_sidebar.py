"""Abstract base class shared by AAboveBSidebar and ProtocolSidebar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.interval import AAboveBInterval
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar


@dataclass
class DepthColumEntrySidebar(Sidebar[DepthColumnEntry]):
    """Abstract base for sidebars whose entries are individual depth values (DepthColumnEntry).

    Shared by AAboveBSidebar and ProtocolSidebar.
    """

    entries: list[DepthColumnEntry]
    skipped_entries: list[DepthColumnEntry] = field(default_factory=list)

    def strictly_contains(self, other: DepthColumEntrySidebar) -> bool:
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_strictly_increasing(self) -> bool:
        return all(self.entries[i].value < self.entries[i + 1].value for i in range(len(self.entries) - 1))

    def depth_intervals(self) -> list[AAboveBInterval]:
        """Creates a list of depth intervals from the depth column entries.

        The first depth interval has an open start value (i.e. None).

        Returns:
            list[AAboveBInterval]: A list of depth intervals.
        """
        depth_intervals = []
        if self.entries[0].value != 0.0:
            depth_intervals.append(AAboveBInterval(None, self.entries[0]))
        for i in range(len(self.entries) - 1):
            depth_intervals.append(AAboveBInterval(self.entries[i], self.entries[i + 1]))
        return depth_intervals

    def break_on_double_descending(self) -> list[Self]:
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

        return [type(self)(segment, self.skipped_entries) for segment in segments]
