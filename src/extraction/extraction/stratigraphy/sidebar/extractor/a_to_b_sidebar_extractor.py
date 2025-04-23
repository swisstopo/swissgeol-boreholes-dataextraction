"""Module for finding AToBSidebar instances in a borehole profile."""

import re

from extraction.extraction.utils.text.textline import TextWord

from ...base.sidebar_entry import DepthColumnEntry
from ...interval.interval import AToBInterval
from ..classes.a_to_b_sidebar import AToBSidebar
from ..utils.cluster import Cluster
from .depth_column_entry_extractor import DepthColumnEntryExtractor


class AToBSidebarExtractor:
    """Class that finds AToBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(all_words: list[TextWord]) -> list[AToBSidebar]:
        """Finds all AToBSidebars.

        Generates a list of AToBDepthColumnEntry objects by finding consecutive pairs of DepthColumnEntry objects.
        Different columns are grouped together in LayerDepthColumn objects. Finally, a list of AToBSidebars objects,
        one for each column, is returned.

        A layer corresponds to a material layer. The layer is defined using a start and end point (e.g. 1.10-1.60m).
        The start and end points are represented as DepthColumnEntry objects.

        Args:
            all_words (list[TextWord]): List of all TextWord objects.

        Returns:
            list[AToBSidebar]: List of all AToBSidebars identified.
        """
        entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=True)

        def find_pair(entry: DepthColumnEntry) -> DepthColumnEntry | None:  # noqa: D103
            min_y0 = entry.rect.y0 - entry.rect.height / 2
            max_y0 = entry.rect.y0 + entry.rect.height / 2
            for other in entries:
                if entry == other:
                    continue
                if other.value <= entry.value:
                    continue
                combined_width = entry.rect.width + other.rect.width
                if not entry.rect.x0 <= other.rect.x0 <= entry.rect.x0 + combined_width:
                    continue
                if not min_y0 <= other.rect.y0 <= max_y0:
                    continue
                in_between_text = " ".join(
                    [
                        word.text
                        for word in all_words
                        if entry.rect.x0 < word.rect.x0 < other.rect.x0 and min_y0 <= word.rect.y0 <= max_y0
                    ]
                )
                if re.fullmatch(r"\W*m?\W*", in_between_text):
                    return other

        pairs = [(entry, find_pair(entry)) for entry in entries]
        intervals = [AToBInterval(first, second) for first, second in pairs if second]
        clusters = Cluster[AToBInterval].create_clusters(intervals, lambda interval: interval.rect)

        return [
            sidebar_segment
            for cluster in clusters
            for sidebar_segment in AToBSidebar(cluster.entries).break_on_mismatch()
            if sidebar_segment.is_valid()
        ]
