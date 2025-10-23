"""Module for the spulprobe sidebars."""

from typing import ClassVar

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.interval.interval import IntervalZone
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

        The interval zones are created from the AToBInterval entries, filtering out sublayers and invalid layers.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        return self.get_zones_from_entries(self.entries, include_open_ended=True)
