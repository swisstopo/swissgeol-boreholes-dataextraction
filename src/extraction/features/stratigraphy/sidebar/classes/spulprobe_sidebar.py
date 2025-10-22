"""Module for the spulprobe sidebars."""

from typing import ClassVar

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.interval.interval import IntervalZone, SpulprobeInterval
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from extraction.features.utils.text.textline import TextLine


class SpulprobeSidebar(Sidebar[SpulprobeEntry]):
    """Spulprobe sidebar where entries are depths in the form `Sp. X m`."""

    kind: ClassVar[str] = "spulprobe"

    @staticmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines to Spulprobe zones.

        The score is 1.0 if the line is located within the interval zone, 0.0 otherwise.
        For Spulprobe sidebar, the zone begins and ends at the top of each rectangle bounds.

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
        zones = [
            IntervalZone(entry.rect, next_entry.rect, SpulprobeInterval(entry, next_entry))
            for entry, next_entry in zip(self.entries, self.entries[1:], strict=False)
        ]
        return zones + [IntervalZone(self.entries[-1], None, SpulprobeInterval(self.entries[-1], None))]
