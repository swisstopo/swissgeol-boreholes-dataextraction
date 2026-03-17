"""Module for finding ProtocolSidebar instances in a borehole profile."""

from __future__ import annotations

import logging

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.depth_column_entry_extractor import DepthColumnEntryExtractor
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import ProtocolSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import SidebarNoise, noise_count
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster_protocol
from swissgeol_doc_processing.geometry.util import x_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine, TextWord

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)


class ProtocolSidebarExtractor:
    """Class that finds ProtocolSidebar instances in a borehole profile."""

    @staticmethod
    def _debug_sidebar(prefix: str, sidebar: ProtocolSidebar) -> None:
        """Log a compact representation of a protocol sidebar candidate."""
        logger.debug(
            "%s | entries=%s | rect=%s",
            prefix,
            [entry.value for entry in sidebar.entries],
            sidebar.rect,
        )

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
        logger.debug("Protocol: raw depth entries after used-entry filtering = %s", len(entries))
        if entries:
            logger.debug("Protocol: raw depth entry values = %s", [entry.value for entry in entries])

        if not entries:
            return []

        clusters = Cluster_protocol[DepthColumnEntry].create_clusters_protocol(entries, lambda entry: entry.rect)
        logger.debug("Protocol: total clusters = %s", len(clusters))
        logger.debug(
            "Protocol: cluster sizes = %s",
            [len(cluster.entries) for cluster in clusters],
        )

        min_entries = sidebar_params.get("min_entries", 2)
        header_keywords = tuple(sidebar_params.get("header_keywords", ("Tiefe",)))
        max_header_gap = sidebar_params.get("max_header_gap", 40)

        candidate_sidebars = [
            ProtocolSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= min_entries
        ]
        logger.debug(
            "Protocol: candidate sidebars with at least %s entries = %s",
            min_entries,
            len(candidate_sidebars),
        )
        for sidebar in candidate_sidebars:
            ProtocolSidebarExtractor._debug_sidebar("Protocol candidate", sidebar)

        processed_sidebars = []
        for sidebar in candidate_sidebars:
            has_header = ProtocolSidebarExtractor._is_below_header(sidebar, lines, header_keywords, max_header_gap)
            is_table_like = ProtocolSidebarExtractor._is_table_like(sidebar, lines)

            if not has_header:
                ProtocolSidebarExtractor._debug_sidebar("Protocol rejected: no header", sidebar)
                continue

            if not is_table_like:
                ProtocolSidebarExtractor._debug_sidebar("Protocol rejected: not table-like", sidebar)
                continue

            processed = sidebar.process()
            if not processed:
                ProtocolSidebarExtractor._debug_sidebar("Protocol rejected: empty after process()", sidebar)
                continue

            valid_processed = [processed_sidebar for processed_sidebar in processed if processed_sidebar.is_valid()]
            if not valid_processed:
                ProtocolSidebarExtractor._debug_sidebar("Protocol rejected: invalid after process()", sidebar)
                continue

            for processed_sidebar in valid_processed:
                ProtocolSidebarExtractor._debug_sidebar("Protocol accepted", processed_sidebar)
                processed_sidebars.append(processed_sidebar)

        logger.debug("Protocol: accepted sidebars = %s", len(processed_sidebars))

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

        logger.debug("Protocol: final sidebars after containment filtering = %s", len(result))
        for sidebar_noise in result:
            ProtocolSidebarExtractor._debug_sidebar("Protocol final", sidebar_noise.sidebar)

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
                logger.debug(
                    "Protocol header candidate rejected: below first entry | header=%r | entries=%s",
                    line.text,
                    [entry.value for entry in sidebar.entries],
                )
                continue
            if first_entry_rect.y0 - line.rect.y1 > max_header_gap:
                logger.debug(
                    "Protocol header candidate rejected: too far away | header=%r | gap=%s | entries=%s",
                    line.text,
                    first_entry_rect.y0 - line.rect.y1,
                    [entry.value for entry in sidebar.entries],
                )
                continue
            if x_overlap_significant_smallest(line.rect, sidebar.rect, 0.2) or line.rect.x0 <= sidebar.rect.x1:
                logger.debug(
                    "Protocol header match | header=%r | entries=%s",
                    line.text,
                    [entry.value for entry in sidebar.entries],
                )
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
        logger.debug(
            "Protocol table-like check | entries=%s | matching_rows=%s | threshold=%s | matched_values=%s",
            [entry.value for entry in sidebar.entries],
            matching_rows,
            threshold,
            matched_values,
        )
        return matching_rows >= threshold
