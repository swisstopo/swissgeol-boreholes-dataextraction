"""Module for finding AAboveBSidebar instances in a borehole profile."""

import logging
from bisect import bisect_left, bisect_right
from collections import defaultdict

import fitz
import rtree

from stratigraphy.depth import DepthColumnEntryExtractor
from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .a_above_b_sidebar_validator import AAboveBSidebarValidator
from .sidebar import SidebarNoise, noise_count

logger = logging.getLogger(__name__)


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(
        all_words: list[TextWord],
        word_rtree: rtree.index.Index,
        used_entry_rects: list[fitz.Rect],
        sidebar_params: dict,
    ) -> list[SidebarNoise]:
        """Construct all possible AAboveBSidebar objects from the given words.

        Args:
            all_words (list[TextWord]): All words in the page.
            word_rtree (rtree.index.Index): Pre-built R-tree for spatial queries.
            used_entry_rects (list[fitz.Rect]): Part of the document to ignore.
            sidebar_params (dict): Parameters for the AAboveBSidebar objects.

        Returns:
            list[SidebarNoise]: Validated AAboveBSidebar objects wrapped with noise count.
        """
        entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]
        clusters = []
        cluster_dict = defaultdict(lambda: AAboveBSidebar(entries=[]))
        threshold = 15

        for entry in entries:
            x0 = entry.rect.x0
            left = bisect_left(clusters, x0 - threshold)
            right = bisect_right(clusters, x0 + threshold)

            matched_clusters = clusters[left:right]

            create_new_cluster = True
            for cluster_x0 in matched_clusters:
                cluster_dict[cluster_x0].entries.append(entry)
                if abs(cluster_x0 - x0) <= threshold / 2:
                    create_new_cluster = False
                    break

            if create_new_cluster:
                clusters.insert(right, x0)
                cluster_dict[x0].entries.append(entry)

        numeric_columns = [cluster for cluster in cluster_dict.values() if len(cluster.entries) > 3]

        filtered_columns = [
            column
            for numeric_column in numeric_columns
            for column in numeric_column.break_on_double_descending()
            if not column.significant_arithmetic_progression()
        ]

        sidebar_validator = AAboveBSidebarValidator(**sidebar_params)

        def process_column(column):
            noise = noise_count(column, all_words, word_rtree)
            sidebar_noise = SidebarNoise(sidebar=column, noise_count=noise)
            return sidebar_validator.reduce_until_valid(sidebar_noise, all_words, word_rtree)

        validated_sidebars = list(filter(None, map(process_column, filtered_columns)))

        return sorted(
            [sidebar_noise for sidebar_noise in validated_sidebars if sidebar_noise.sidebar],
            key=lambda sidebar_noise: len(sidebar_noise.sidebar.entries),
        )
