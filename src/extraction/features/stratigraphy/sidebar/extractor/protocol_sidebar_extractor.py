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

        min_entries = sidebar_params.get("min_entries", 2)
        header_keywords = tuple(sidebar_params.get("header_keywords", ("Tiefe",)))
        max_header_gap = sidebar_params.get("max_header_gap", 40)

        candidate_sidebars = [
            ProtocolSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= min_entries
        ]

        processed_sidebars = []
        for sidebar in candidate_sidebars:
            has_header = ProtocolSidebarExtractor._is_below_header(sidebar, lines, header_keywords, max_header_gap)
            is_table_like = ProtocolSidebarExtractor._is_table_like(sidebar, lines)

            if not has_header:
                continue

            if not is_table_like:
                continue

            processed = sidebar.process()
            if not processed:
                continue

            valid_processed = [processed_sidebar for processed_sidebar in processed if processed_sidebar.is_valid()]
            if not valid_processed:
                continue

            for processed_sidebar in valid_processed:
                processed_sidebars.append(processed_sidebar)

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
        lines: list[TextLine],
        header_keywords: tuple[str, ...],
        max_header_gap: float,
    ) -> bool:
        """Check whether the sidebar is located below a matching protocol header."""
        first_entry_rect = sidebar.entries[0].rect
        normalized_keywords = {keyword.casefold() for keyword in header_keywords}

        for line in lines:
            line_text = line.text.casefold()
            if not any(keyword in line_text for keyword in normalized_keywords):
                continue
            if line.rect.y1 > first_entry_rect.y0:
                continue
            if first_entry_rect.y0 - line.rect.y1 > max_header_gap:
                continue
            if x_overlap_significant_smallest(line.rect, sidebar.rect, 0.2) or line.rect.x0 <= sidebar.rect.x1:
                return True

        return False

    @staticmethod
    def _is_table_like(sidebar: ProtocolSidebar, lines: list[TextLine]) -> bool:
        """Check whether the sidebar sits inside a simple table-like structure."""
        matching_rows = 0
        matched_values = []

        for entry in sidebar.entries:
            for line in lines:
                if line.rect.x0 <= entry.rect.x1:
                    continue
                if abs(line.rect.y0 - entry.rect.y0) > max(entry.rect.height, line.rect.height):
                    continue
                if not any(char.isalnum() for char in line.text):
                    continue
                matching_rows += 1
                matched_values.append(entry.value)
                break

        threshold = max(1, len(sidebar.entries) // 2)

        return matching_rows >= threshold
