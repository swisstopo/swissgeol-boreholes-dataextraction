"""Module for finding AAboveBSidebar instances in a borehole profile."""

import statistics

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.depth_column_entry_extractor import DepthColumnEntryExtractor
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import AAboveBSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import SidebarNoise, noise_count
from extraction.features.stratigraphy.sidebar.utils.a_above_b_sidebar_validator import AAboveBSidebarValidator
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from swissgeol_doc_processing.text.textline import TextWord


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def _arithmetic_progression_entries(entries: list[DepthColumnEntry]) -> list[DepthColumnEntry]:
        # First, try to find an artihmetic progression in the entries without decimal point
        no_decimal_point_entries = [entry for entry in entries if not entry.has_decimal_point]
        no_decimal_point_values = [entry.value for entry in no_decimal_point_entries]
        selected_values = AAboveBSidebarExtractor._arithmetic_progression_values(no_decimal_point_values)
        selected_entries = [entry for entry in no_decimal_point_entries if entry.value in selected_values]
        if selected_entries:
            return selected_entries

        # If nothing found in the integer entries, then retry with all entries
        values = [entry.value for entry in entries]
        selected_values = AAboveBSidebarExtractor._arithmetic_progression_values(values)
        return [entry for entry in entries if entry.value in selected_values]

    @staticmethod
    def _arithmetic_progression_values(values: list[float]) -> set[float]:
        """Check if some of the values form an arithmetic progression."""
        if len(values) <= 2:
            return {}

        integer_values = [int(round(value * 100)) for value in values]
        differences = [integer_values[i + 1] - integer_values[i] for i in range(len(integer_values) - 1)]
        step = statistics.mode(differences)
        if step <= 0:
            return {}

        # only consider arithmetic progressions that include 0 (when extended if necessary)
        candidate_values = [value for value in integer_values if value % step == 0]

        values_set = set(integer_values)
        matching_steps = [value + step in values_set for value in candidate_values].count(True)
        # For at least 70% of all values (except the highest one), the adding the step should give another present
        # value.
        if matching_steps > 0.7 * (len(integer_values) - 1):
            # return candidate values that are part of a segment of at least 3 consecutive values
            segments = [[value - step, value, value + step] for value in candidate_values]
            return {
                value / 100
                for segment in segments
                if all(value in values_set for value in segment)
                for value in segment
            }
        else:
            return {}

    @staticmethod
    def find_in_words(
        all_words: list[TextWord],
        line_rtree: fastquadtree.RectQuadTreeObjects,
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
            if all((entry.rect & used_rect).is_empty for used_rect in used_entry_rects)
        ]
        clusters = Cluster[DepthColumnEntry].create_clusters(entries, lambda entry: entry.rect)

        excluded_entries = {
            entry
            for cluster in clusters
            for entry in AAboveBSidebarExtractor._arithmetic_progression_entries(cluster.entries)
        }

        if excluded_entries:
            # cluster again, but without the entries that are part of an arithmetic progression
            entries = [entry for entry in entries if entry not in excluded_entries]
            clusters = Cluster[DepthColumnEntry].create_clusters(entries, lambda entry: entry.rect)

        numeric_columns = [AAboveBSidebar(cluster.entries) for cluster in clusters]

        filtered_columns = [
            column
            for numeric_column in numeric_columns
            for column in numeric_column.fix_ocr_mistakes().break_on_double_descending()
            if len(column.entries) >= 3
        ]

        sidebar_validator = AAboveBSidebarValidator(**sidebar_params)

        def process_column(column):
            noise = noise_count(column, line_rtree)
            sidebar_noise = SidebarNoise(sidebar=column, noise_count=noise)
            return sidebar_validator.reduce_until_valid(sidebar_noise, line_rtree)

        validated_sidebars = list(filter(None, map(process_column, filtered_columns)))

        sidebars_by_length = sorted(
            [sidebar_noise for sidebar_noise in validated_sidebars if sidebar_noise.sidebar],
            key=lambda sidebar_noise: len(sidebar_noise.sidebar.entries),
            reverse=True,
        )

        result = []
        # Remove sidebar_noise that are fully contained in a (strictly) longer sidebar
        for sidebar_noise in sidebars_by_length:
            if not any(
                (
                    result_sidebar.sidebar.rect.contains(sidebar_noise.sidebar.rect)
                    and len(result_sidebar.sidebar.entries) > len(sidebar_noise.sidebar.entries)
                )
                for result_sidebar in result
            ):
                result.append(sidebar_noise)

        return result
