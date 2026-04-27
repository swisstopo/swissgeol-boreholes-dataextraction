"""Module for finding ProtocolSidebar instances in a borehole profile."""

from __future__ import annotations

import statistics

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.depth_column_entry_extractor import DepthColumnEntryExtractor
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import ProtocolSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import SidebarNoise, noise_count
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from swissgeol_doc_processing.geometry.util import x_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from swissgeol_doc_processing.utils.table_detection import TableStructure


class ProtocolSidebarExtractor:
    """Class that finds ProtocolSidebar instances in a borehole profile."""

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

        candidate_values = [value for value in integer_values if value % step == 0]

        values_set = set(integer_values)
        matching_steps = [value + step in values_set for value in candidate_values].count(True)
        if matching_steps > 0.7 * (len(integer_values) - 1):
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
    def _arithmetic_progression_entries(entries: list[DepthColumnEntry]) -> list[DepthColumnEntry]:
        no_decimal_point_entries = [entry for entry in entries if not entry.has_decimal_point]
        no_decimal_point_values = [entry.value for entry in no_decimal_point_entries]
        selected_values = ProtocolSidebarExtractor._arithmetic_progression_values(no_decimal_point_values)
        selected_entries = [entry for entry in no_decimal_point_entries if entry.value in selected_values]
        if selected_entries:
            return selected_entries

        values = [entry.value for entry in entries]
        selected_values = ProtocolSidebarExtractor._arithmetic_progression_values(values)
        return [entry for entry in entries if entry.value in selected_values]

    @staticmethod
    def find_in_words(
        all_words: list[TextWord],
        lines: list[TextLine],
        line_rtree: fastquadtree.RectQuadTreeObjects,
        used_entry_rects: list[pymupdf.Rect],
        table_structures: list[TableStructure],
        sidebar_params: dict,
    ) -> list[SidebarNoise]:
        """Construct all possible ProtocolSidebar objects from the given words.

        Args:
            all_words (list[TextWord]): All words in the page.
            lines (list[TextLine]): All text lines in the page.
            line_rtree (fastquadtree.RectQuadTreeObjects): Pre-built R-tree for spatial queries.
            used_entry_rects (list[pymupdf.Rect]): Part of the document to ignore.
            table_structures: list[TableStructure]:  List of table structures.
            sidebar_params (dict): Parameters for the ProtocolSidebar objects.

        Returns:
            list[SidebarNoise]: Valid ProtocolSidebar objects wrapped with noise count.
        """
        min_entries = sidebar_params.get("min_entries")
        header_keywords = sidebar_params.get("header_keywords")
        max_header_gap = sidebar_params.get("max_header_gap")
        header_x_expansion = sidebar_params.get("header_x_expansion", 20)

        normalized_keywords = {keyword.casefold() for keyword in header_keywords}
        header_lines = [
            line for line in lines if any(keyword in line.text.casefold() for keyword in normalized_keywords)
        ]

        if not header_lines:
            return []

        # Extract all candidate depth entries not already claimed by other sidebar types
        all_entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words)
            if all((entry.rect & used_rect).is_empty for used_rect in used_entry_rects)
        ]

        processed_sidebars = []
        seen_entry_sets = []  # avoid duplicate sidebars from overlapping headers

        for header_line in header_lines:
            expanded_header_rect = pymupdf.Rect(
                header_line.rect.x0 - header_x_expansion,
                header_line.rect.y0,
                header_line.rect.x1 + header_x_expansion,
                header_line.rect.y1,
            )
            # Only entries that are below this header and x-aligned with the expanded rect
            column_entries = [
                entry
                for entry in all_entries
                if entry.rect.y0 >= header_line.rect.y1 - max_header_gap
                and x_overlap_significant_smallest(entry.rect, expanded_header_rect, 0.2)
            ]

            if not column_entries:
                continue

            clusters = Cluster[DepthColumnEntry].create_clusters(
                column_entries, lambda entry: entry.rect, allow_size_two=True
            )

            excluded_entries = {
                entry
                for cluster in clusters
                for entry in ProtocolSidebarExtractor._arithmetic_progression_entries(cluster.entries)
            }
            if excluded_entries:
                column_entries = [e for e in column_entries if e not in excluded_entries]
                clusters = Cluster[DepthColumnEntry].create_clusters(
                    column_entries, lambda entry: entry.rect, allow_size_two=True
                )

            for cluster in clusters:
                if len(cluster.entries) < min_entries:
                    continue

                sidebar = ProtocolSidebar(cluster.entries)

                # Skip duplicate clusters already found via another header
                entry_ids = frozenset(id(e) for e in sidebar.entries)
                if any(entry_ids == seen for seen in seen_entry_sets):
                    continue

                is_in_table = any(
                    table_structure.bounding_rect.contains(sidebar.rect) for table_structure in table_structures
                )
                if not is_in_table:
                    continue

                processed = sidebar.process()
                if not processed:
                    continue

                seen_entry_sets.append(entry_ids)
                processed_sidebars.extend(processed)

        sidebars_with_noise = [
            SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, line_rtree))
            for sidebar in processed_sidebars
        ]

        sidebars_by_length = sorted(
            sidebars_with_noise,
            key=lambda sidebar_noise: len(sidebar_noise.sidebar.entries),
            reverse=True,
        )

        result = []
        for sidebar_noise in sidebars_by_length:
            if not any(result_sidebar.sidebar.strictly_contains(sidebar_noise.sidebar) for result_sidebar in result):
                result.append(sidebar_noise)

        return result
