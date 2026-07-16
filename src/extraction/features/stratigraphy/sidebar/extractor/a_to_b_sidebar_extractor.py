"""Module for finding AToBSidebar instances in a borehole profile."""

import re

from extraction.features.stratigraphy.interval.a_to_b_interval_extractor import AToBIntervalExtractor
from extraction.features.stratigraphy.interval.depth_column_entry_extractor import DepthColumnEntryExtractor
from extraction.features.stratigraphy.interval.interval import AToBInterval
from extraction.features.stratigraphy.sidebar.classes.a_to_b_sidebar import AToBSidebar
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from extraction.features.stratigraphy.sidebar.utils.entries_per_table import TableEntries
from extraction.features.stratigraphy.sidebarentry.depth_column_entry import DepthColumnEntry
from extraction.features.stratigraphy.sidebarentry.interval_entry import IntervalEntry
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from swissgeol_doc_processing.utils.table_detection import TableStructure


class AToBSidebarExtractor:
    """Class that finds AToBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(all_words: list[TextWord], table_structures: list[TableStructure]) -> list[AToBSidebar]:
        """Finds all AToBSidebars, where start and end point of a layer are defined together (e.g. 1.10-1.60m).

        Generates a list of IntervalEntry objects by finding consecutive pairs of DepthColumnEntry objects.
        Intervals are clustered together into columns. Finally, a list of AToBSidebars objects, one for each
        column, is returned.

        The start and end points are represented as DepthColumnEntry objects.

        Args:
            all_words (list[TextWord]): List of all TextWord objects.
            table_structures (list[TableStructure]): List of detected table-like structures

        Returns:
            list[AToBSidebar]: List of all AToBSidebars identified.
        """
        interval_entries: list[IntervalEntry] = []
        for word in all_words:
            a_to_b_interval, _ = AToBIntervalExtractor.from_text(TextLine([word]))
            if (
                a_to_b_interval
                and a_to_b_interval.start
                and a_to_b_interval.end
                and (a_to_b_interval.start.value < a_to_b_interval.end.value)
            ):
                interval_entries.append(IntervalEntry(a_to_b_interval, word.page_number))

        # Find additional pairs that do not come from a single TextWord
        entries = DepthColumnEntryExtractor.find_in_words(all_words)

        def find_pair(entry: DepthColumnEntry) -> DepthColumnEntry | None:  # noqa: D103
            min_y0 = entry.rect.y0 - entry.rect.height / 2
            max_y0 = entry.rect.y0 + entry.rect.height / 2
            min_x0 = max(entry.rect.x0, entry.rect.x1 - entry.rect.height)
            for other in entries:
                if entry == other:
                    continue
                if other.value <= entry.value:
                    continue
                combined_width = entry.rect.width + other.rect.width
                if not min_x0 <= other.rect.x0 <= entry.rect.x0 + combined_width:
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

        for entry in entries:
            other = find_pair(entry)
            if other:
                interval = AToBInterval(entry, other)
                interval_entries.append(IntervalEntry(interval, entry.page_number))

        entry_partitions = TableEntries.group_entries_by_table(
            table_structures, sorted(interval_entries, key=lambda entry: entry.rect.y0)
        )
        clusters = [
            cluster
            for partition in entry_partitions
            for cluster in Cluster[IntervalEntry].create_clusters(
                partition.entries,
                table_structure=partition.table,
                allow_size_two=True,
            )
        ]
        return [
            sidebar_segment
            for cluster in clusters
            for sidebar_segment in AToBSidebar(cluster.entries).process()
            if sidebar_segment.is_valid()
        ]
