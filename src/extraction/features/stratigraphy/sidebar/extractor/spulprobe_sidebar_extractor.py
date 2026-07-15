"""Module for the extraction of Sidebars coming from Sp. sampled entries."""

import re

import pymupdf

from extraction.features.stratigraphy.sidebar.classes.spulprobe_sidebar import SpulprobeSidebar
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from extraction.features.stratigraphy.sidebar.utils.entries_per_table import TableEntries
from extraction.features.stratigraphy.sidebarentry.depth_column_entry import DepthColumnEntry
from swissgeol_doc_processing.geometry.util import compute_outer_rect, y_overlap_significant_smallest
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.table_detection import TableStructure


class SpulprobeSidebarExtractor:
    """Spulprobe sidebar extractor, in charge of identifying Sp. tags, and extracting the associate depth."""

    spulprobe_pattern = r"\bSp\.?"
    depths_pattern = r"\d+(?:[.,]\d+)?"

    @classmethod
    def find_spulprobe_entries(cls, lines: list[TextLine]) -> list[DepthColumnEntry]:
        """Find the spulprobe entries.

        Args:
            lines (list[TextLine]): The text lines to search in.

        Returns:
            list[DepthColumnEntry]: A list of DepthColumnEntry objects found in the lines.
        """
        entries = []
        for line in sorted(lines, key=lambda line: (line.rect.y0, line.rect.x0)):
            if len(line.words) > 0:
                # Only match in the first word of every line.
                first_word = line.words[0]
                regex = re.compile(cls.spulprobe_pattern)
                match = regex.match(first_word.text)
                if not match:
                    continue
                depths = [float(m.replace(",", ".")) for m in re.findall(cls.depths_pattern, line.text)]
                entry_rect = line.rect
                if not depths:
                    depths, line_rect = cls.search_depths_in_lines_on_the_right(line, lines)
                    if not depths:
                        continue
                    entry_rect = compute_outer_rect([line.rect, line_rect])
                page_number = line.page_number
                most_shallow = min(depths)
                entries.append(DepthColumnEntry(rect=entry_rect, value=most_shallow, page_number=page_number))
        return entries

    @classmethod
    def search_depths_in_lines_on_the_right(
        cls, current_line: TextLine, lines: list[TextLine]
    ) -> tuple[list[float], pymupdf.Rect | None]:
        """Searches for depths in lines that are to the right of the line with the Sp. tag.

        Args:
            current_line (TextLine): The current line where the Sp. tag was identified.
            lines (list[TextLine]): The list of lines to search in.

        Returns:
            tuple[list[float], pymupdf.Rect | None] : A tuple containing a list of depths found and the rectangle of
                the line where they were found.
        """
        for line in lines:
            if line == current_line:
                continue
            if not y_overlap_significant_smallest(current_line.rect, line.rect, 0.9):
                continue
            if line.rect.x0 < current_line.rect.x1:
                continue
            return [float(m.replace(",", ".")) for m in re.findall(cls.depths_pattern, line.text)], line.rect
        return [], None

    @classmethod
    def find_in_lines(cls, lines: list[TextLine], table_structures: list[TableStructure]) -> list[SpulprobeSidebar]:
        """Find Spulprobe sidebars in the given lines.

        Args:
            lines (list[TextLine]): The text lines to search in.
            table_structures (list[TableStructure]): List of detected table-like structures

        Returns:
            list[SpulprobeSidebar]: A list of SpulprobeSidebar objects found in the lines.
        """
        entries = cls.find_spulprobe_entries(lines)

        entry_partitions = TableEntries.group_entries_by_table(table_structures, entries)
        clusters = [
            cluster
            for partition in entry_partitions
            for cluster in Cluster[DepthColumnEntry].create_clusters(
                partition.entries, table_structure=partition.table
            )
        ]

        return [SpulprobeSidebar(cluster.entries) for cluster in clusters]
