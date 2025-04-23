"""Module for finding AAboveBSidebar instances in a borehole profile."""

import pymupdf
import rtree
from borehole_extraction.extraction.util_extraction.text.textline import TextWord

from ..base_sidebar_entry.sidebar_entry import DepthColumnEntry
from ..sidebar_classes.a_above_b_sidebar import AAboveBSidebar
from ..sidebar_classes.sidebar import SidebarNoise, noise_count
from ..sidebar_extractor.depth_column_entry_extractor import DepthColumnEntryExtractor
from ..sidebar_utils.a_above_b_sidebar_validator import AAboveBSidebarValidator
from ..sidebar_utils.cluster import Cluster


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(
        all_words: list[TextWord],
        line_rtree: rtree.index.Index,
        used_entry_rects: list[pymupdf.Rect],
        sidebar_params: dict,
    ) -> list[SidebarNoise]:
        """Construct all possible AAboveBSidebar objects from the given words.

        Args:
            all_words (list[TextWord]): All words in the page.
            line_rtree (rtree.index.Index): Pre-built R-tree for spatial queries.
            used_entry_rects (list[pymupdf.Rect]): Part of the document to ignore.
            sidebar_params (dict): Parameters for the AAboveBSidebar objects.

        Returns:
            list[SidebarNoise]: Validated AAboveBSidebar objects wrapped with noise count.
        """
        entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]
        clusters = Cluster[DepthColumnEntry].create_clusters(entries, lambda entry: entry.rect)

        numeric_columns = [AAboveBSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= 3]

        filtered_columns = [
            column
            for numeric_column in numeric_columns
            for column in numeric_column.make_ascending().break_on_double_descending()
            if not column.close_to_arithmetic_progression()
        ]

        sidebar_validator = AAboveBSidebarValidator(**sidebar_params)

        def process_column(column):
            noise = noise_count(column, line_rtree)
            sidebar_noise = SidebarNoise(sidebar=column, noise_count=noise)
            return sidebar_validator.reduce_until_valid(sidebar_noise, line_rtree)

        validated_sidebars = list(filter(None, map(process_column, filtered_columns)))

        for sidebar_noise in validated_sidebars:
            sidebar_noise.sidebar.remove_integer_scale()

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
