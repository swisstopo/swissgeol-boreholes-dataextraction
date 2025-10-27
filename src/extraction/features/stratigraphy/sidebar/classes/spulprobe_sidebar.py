"""Module for the spulprobe sidebars."""

from typing import ClassVar

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.interval.interval import Interval, IntervalZone
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
        return Sidebar.default_score(interval_zone, line)

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        The interval zones are created from the Spulprobe entries, including open-ended intervals at both ends.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        open_started = IntervalZone(None, self.entries[0].rect, Interval(None, self.entries[0]))
        zones = [
            IntervalZone(entry.rect, next_entry.rect, Interval(entry, next_entry))
            for entry, next_entry in zip(self.entries, self.entries[1:], strict=False)
        ]
        open_ended = IntervalZone(self.entries[-1].rect, None, Interval(self.entries[-1], None))
        return [open_started] + zones + [open_ended]
