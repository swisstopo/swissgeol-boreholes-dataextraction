"""Module for finding ProtocolSidebar instances in a borehole profile."""

from __future__ import annotations

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.depth_column_entry_extractor import DepthColumnEntryExtractor
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import ProtocolSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import SidebarNoise, noise_count
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from swissgeol_doc_processing.geometry.util import x_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine, TextWord


class ProtocolSidebarExtractor:
    """Class that finds ProtocolSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(
        all_words: list[TextWord],
        lines: list[TextLine],
        line_rtree: fastquadtree.RectQuadTreeObjects,
        used_entry_rects: list[pymupdf.Rect],
        sidebar_params: dict,
    ) -> list[SidebarNoise]:
        """Construct all possible ProtocolSidebar objects from the given words.

        Args:
            all_words (list[TextWord]): All words in the page.
            lines (list[TextLine]): All text lines in the page.
            line_rtree (fastquadtree.RectQuadTreeObjects): Pre-built R-tree for spatial queries.
            used_entry_rects (list[pymupdf.Rect]): Part of the document to ignore.
            sidebar_params (dict): Parameters for the ProtocolSidebar objects.

        Returns:
            list[SidebarNoise]: Valid ProtocolSidebar objects wrapped with noise count.
        """
        entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words)
            if all((entry.rect & used_rect).is_empty for used_rect in used_entry_rects)
        ]

        if not entries:
            return []

        clusters = Cluster[DepthColumnEntry].create_clusters(entries, lambda entry: entry.rect, allow_size_two=True)

        min_entries = sidebar_params.get("min_entries")
        header_keywords = sidebar_params.get("header_keywords")
        max_header_gap = sidebar_params.get("max_header_gap")
        normalized_keywords = {keyword.casefold() for keyword in header_keywords}
        header_lines = [
            line for line in lines if any(keyword in line.text.casefold() for keyword in normalized_keywords)
        ]

        candidate_sidebars = [
            ProtocolSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= min_entries
        ]

        processed_sidebars = []
        for sidebar in candidate_sidebars:
            has_header = ProtocolSidebarExtractor._is_below_header(sidebar, header_lines, max_header_gap)
            is_table_like = ProtocolSidebarExtractor._is_table_like(sidebar, lines)

            if not has_header:
                continue

            if not is_table_like:
                continue

            processed = sidebar.process()
            if not processed:
                continue

            valid_processed = [
                processed_sidebar
                for processed_sidebar in processed
                if ProtocolSidebarExtractor._has_material_match_for_each_entry(
                    processed_sidebar,
                    lines,
                )
            ]

            processed_sidebars.extend(valid_processed)

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

    @staticmethod
    def _is_below_header(
        sidebar: ProtocolSidebar,
        header_lines: list[TextLine],
        max_header_gap: float,
    ) -> bool:
        """Check whether the sidebar is located below a matching protocol header."""
        first_entry_rect = sidebar.entries[0].rect
        for line in header_lines:
            if line.rect.y1 > first_entry_rect.y0:
                continue
            if 0 <= first_entry_rect.y0 - line.rect.y1 <= max_header_gap and x_overlap_significant_smallest(
                line.rect, sidebar.rect, 0.2
            ):
                return True
        return False

    @staticmethod
    def _has_material_match_for_each_entry(
        sidebar: ProtocolSidebar,
        lines: list[TextLine],
    ) -> bool:
        """Check whether each depth entry has a corresponding material-description line.

        A valid match is defined as:
        - a text line containing at least one alphanumeric character
        - the line is located to the right of the depth entry
        - the line is roughly aligned vertically with the entry (same row)
        - each line can only be matched to one entry
        """
        # Filter lines to those that contain actual text
        candidate_lines = [line for line in lines if any(char.isalnum() for char in line.text)]

        used_line_indices: set[int] = set()

        # Find a matching description line for each depth entry
        for entry in sidebar.entries:
            match_found = False

            for index, line in enumerate(candidate_lines):
                # Skip lines that are already matched to another entry
                if index in used_line_indices:
                    continue

                # The description must be to the right of the depth column
                if line.rect.x0 <= entry.rect.x1:
                    continue

                # Check vertical alignment (approximate "same row")
                if abs(line.rect.y0 - entry.rect.y0) > max(entry.rect.height, line.rect.height):
                    continue

                used_line_indices.add(index)
                match_found = True
                break

            if not match_found:
                return False

        return True

    @staticmethod
    def _is_table_like(sidebar: ProtocolSidebar, lines: list[TextLine]) -> bool:
        """Check whether the sidebar sits inside a simple table-like structure."""
        matching_rows = 0

        for entry in sidebar.entries:
            for line in lines:
                if line.rect.x0 <= entry.rect.x1:
                    continue
                if abs(line.rect.y0 - entry.rect.y0) > max(entry.rect.height, line.rect.height):
                    continue
                if not any(char.isalnum() for char in line.text):
                    continue
                matching_rows += 1
                break

        threshold = max(1, len(sidebar.entries) // 2)

        return matching_rows >= threshold
