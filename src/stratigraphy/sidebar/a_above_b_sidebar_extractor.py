"""Module for finding AAboveBSidebar instances in a borehole profile."""

import logging
from collections import defaultdict

import fitz

from stratigraphy.depth import DepthColumnEntryExtractor
from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .a_above_b_sidebar_validator import AAboveBSidebarValidator

logger = logging.getLogger(__name__)


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(  ## help find correct sidebar based on vertical geometric lines!
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
        entries = [  ##takes sometime
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]
        logger.info("entries %s", entries)

        clusters = defaultdict(lambda: AAboveBSidebar(entries=[]))
        for entry in entries:
            x0 = entry.rect.x0
            matched_cluster = []

            for cluster_x0 in clusters:
                if abs(cluster_x0 - x0) <= 10:  ##TODO: edge cases + overlaps handling should be implemented,
                    # set the threshold based on page or entry dimensions?
                    # -> done by allowing entry to belong to multiple clusters ->
                    # use bisect? to reduce time?
                    matched_cluster.append(cluster_x0)

            if matched_cluster:
                for cluster_x0 in matched_cluster:
                    clusters[cluster_x0].entries.append(entry)
            else:
                clusters[x0].entries.append(entry)
        numeric_columns = list(clusters.values())

        logger.info("numeric clusters %s", numeric_columns)

        sidebar_validator = AAboveBSidebarValidator(all_words, **sidebar_params)

        numeric_columns = [
            sidebar_validator.reduce_until_valid(column)
            for numeric_column in numeric_columns
            for column in numeric_column.break_on_double_descending()
            # when we have a perfect arithmetic progression, this is usually just a scale
            # that does not match the descriptions
            if not column.significant_arithmetic_progression()
        ]

        logger.info("numeric_columns valid %s", numeric_columns)

        for column in numeric_columns:  ## create helper function
            if column:
                integer_entries = [entry for entry in column.entries if isinstance(entry.value, int)]
                if integer_entries:
                    integer_subset = AAboveBSidebar(integer_entries)
                    if integer_subset.significant_arithmetic_progression():
                        column.entries = [entry for entry in column.entries if entry not in integer_entries]
        logger.info("numeric_columns filtered out integers%s", numeric_columns)
        return sorted(
            [column for column in numeric_columns if column and sidebar_validator.is_valid(column)],
            key=lambda column: len(column.entries),
        )
