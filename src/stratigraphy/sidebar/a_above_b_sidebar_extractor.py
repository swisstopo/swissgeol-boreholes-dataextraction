"""Module for finding AAboveBSidebar instances in a borehole profile."""

import fitz
import rtree

from stratigraphy.depth import DepthColumnEntryExtractor
from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .a_above_b_sidebar_validator import AAboveBSidebarValidator
from .cluster import Cluster
from .sidebar import SidebarNoise, noise_count
from .sidebarentry import DepthColumnEntry


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
        clusters = Cluster[DepthColumnEntry].create_clusters(entries)

        numeric_columns = [AAboveBSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= 3]

        filtered_columns = [
            column
            for numeric_column in numeric_columns
            for column in numeric_column.make_ascending().break_on_double_descending()
            if not column.close_to_arithmetic_progression()
        ]

        sidebar_validator = AAboveBSidebarValidator(**sidebar_params)

        def process_column(column):
            noise = noise_count(column, word_rtree)
            sidebar_noise = SidebarNoise(sidebar=column, noise_count=noise)
            return sidebar_validator.reduce_until_valid(sidebar_noise, word_rtree)

        validated_sidebars = list(filter(None, map(process_column, filtered_columns)))

        sidebars_by_length = sorted(
            [sidebar_noise for sidebar_noise in validated_sidebars if sidebar_noise.sidebar],
            key=lambda sidebar_noise: len(sidebar_noise.sidebar.entries),
            reverse=True,
        )

        result = []
        # Remove sidebar_noise that are fully contained in a longer sidebar
        for sidebar_noise in sidebars_by_length:
            if not any(
                result_sidebar.sidebar.rect().contains(sidebar_noise.sidebar.rect()) for result_sidebar in result
            ):
                result.append(sidebar_noise)

        return result
