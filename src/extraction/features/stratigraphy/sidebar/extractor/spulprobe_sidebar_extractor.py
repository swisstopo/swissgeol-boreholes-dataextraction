"""Module for the extraction of Sidebars comming from Sp. sampled entries."""

import re

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.sidebar.classes.spulprobe_sidebar import SpulprobeSidebar
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from extraction.features.utils.geometry.util import compute_outer_rect, y_overlap_significant_smallest
from extraction.features.utils.text.textline import TextLine


class SpulprobeSidebarExtractor:
    """spul."""

    spulprobe_pattern = r"\bSp\.?"
    depths_pattern = r"\d+(?:[.,]\d+)?"

    @classmethod
    def find_spulprobe_entries(cls, lines: list[TextLine]) -> list[SpulprobeEntry]:
        """Find the spul."""
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
                deepest = max(depths)
                entries.append(SpulprobeEntry(rect=entry_rect, value=deepest))
        return entries

    @classmethod
    def search_depths_in_lines_on_the_right(cls, current_line: TextLine, lines: list[TextLine]):
        for line in lines:
            if line == current_line:
                continue

            # if abs(line.rect.y0 - current_line.rect.y0) < tol and abs(line.rect.y1 - current_line.rect.y1) < tol:
            if not y_overlap_significant_smallest(current_line.rect, line.rect, 0.9):
                continue
            if line.rect.x0 < current_line.rect.x1:
                continue
            return [float(m.replace(",", ".")) for m in re.findall(cls.depths_pattern, line.text)], line.rect
        return [], None

    @classmethod
    def find_in_lines(cls, lines: list[TextLine]) -> list[SpulprobeSidebar]:
        """Finds all ..."""
        entries = cls.find_spulprobe_entries(lines)

        clusters = Cluster[SpulprobeEntry].create_clusters(entries, lambda entry: entry.rect)

        return [SpulprobeSidebar(cluster.entries) for cluster in clusters]
