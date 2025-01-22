"""Module for finding AAboveBSidebar instances in a borehole profile."""

import fitz

from stratigraphy.depth import DepthColumnEntryExtractor
from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .a_above_b_sidebar_validator import AAboveBSidebarValidator
from .cluster import Cluster
from .sidebarentry import DepthColumnEntry


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(
        all_words: list[TextWord], used_entry_rects: list[fitz.Rect], sidebar_params: dict
    ) -> list[AAboveBSidebar]:
        """Construct all possible AAboveBSidebar objects from the given words.

        Args:
            all_words (list[TextWord]): All words in the page.
            used_entry_rects (list[fitz.Rect]): Part of the document to ignore.
            sidebar_params (dict): Parameters for the AAboveBSidebar objects.

        Returns:
            list[AAboveBSidebar]: Found AAboveBSidebar objects.
        """
        entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]
        clusters = Cluster[DepthColumnEntry].create_clusters(entries)

        numeric_columns = [AAboveBSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= 3]
        sidebar_validator = AAboveBSidebarValidator(all_words, **sidebar_params)

        filtered_columns = [
            column
            for numeric_column in numeric_columns
            for column in numeric_column.make_ascending().break_on_double_descending()
            if not column.close_to_arithmetic_progression()
        ]

        validated_sidebars = [sidebar_validator.reduce_until_valid(column) for column in filtered_columns]

        sidebars_by_length = sorted(
            [sidebar for sidebar in validated_sidebars if sidebar],
            key=lambda sidebar: len(sidebar.entries),
            reverse=True,
        )

        result = []
        # Remove columns that are fully contained in a longer column
        for sidebar in sidebars_by_length:
            if not any(result_sidebar.rect().contains(sidebar.rect()) for result_sidebar in result):
                result.append(sidebar)

        return result
