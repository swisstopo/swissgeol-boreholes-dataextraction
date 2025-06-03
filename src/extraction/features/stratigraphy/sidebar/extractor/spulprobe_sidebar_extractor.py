"""Module for the extraction of Sidebars comming from Sp. sampled entries."""

import re

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.sidebar.classes.spulprobe_sidebar import SpulprobeSidebar
from extraction.features.stratigraphy.sidebar.utils.cluster import Cluster
from extraction.features.utils.text.textline import TextLine


class SpulprobeSidebarExtractor:
    """spul."""

    spulprobe_pattern = r"\bSp\.?"

    @classmethod
    def find_spulprobe_entries(cls, lines: list[TextLine]) -> list[SpulprobeEntry]:
        """Find the spul."""
        entries = []
        for line in sorted(lines, key=lambda line: line.rect.y0):
            if len(line.words) > 0:
                # Only match in the first word of every line.
                first_word = line.words[0]
                regex = re.compile(cls.spulprobe_pattern)
                match = regex.match(first_word.text)
                if not match:
                    continue
                depths = re.findall(r"\d+", line.text)
                if not depths:
                    continue
                deepest = max([float(depth) for depth in depths])
                entries.append(SpulprobeEntry(rect=line.rect, value=deepest))
        return entries

    @classmethod
    def find_in_lines(cls, lines: list[TextLine]) -> list[SpulprobeSidebar]:
        """Finds all ..."""
        entries = cls.find_spulprobe_entries(lines)

        clusters = Cluster[SpulprobeEntry].create_clusters(entries, lambda entry: entry.rect)

        return [SpulprobeSidebar(cluster.entries) for cluster in clusters]
