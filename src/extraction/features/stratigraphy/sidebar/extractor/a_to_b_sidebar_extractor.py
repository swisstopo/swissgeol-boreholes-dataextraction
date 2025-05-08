"""Module for finding AToBSidebar instances in a borehole profile."""

import re

from extraction.features.utils.text.textline import TextWord

from ...base.sidebar_entry import DepthColumnEntry
from ...interval.interval import AToBInterval
from ..classes.a_to_b_sidebar import AToBSidebar
from ..utils.cluster import Cluster
from .depth_column_entry_extractor import DepthColumnEntryExtractor


class AToBSidebarExtractor:
    """Class that finds AToBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(all_words: list[TextWord]) -> list[AToBSidebar]:
        """Finds all AToBSidebars.

        Generates a list of AToBDepthColumnEntry objects by finding consecutive pairs of DepthColumnEntry objects.
        Different columns are grouped together in LayerDepthColumn objects. Finally, a list of AToBSidebars objects,
        one for each column, is returned.

        A layer corresponds to a material layer. The layer is defined using a start and end point (e.g. 1.10-1.60m).
        The start and end points are represented as DepthColumnEntry objects.

        Args:
            all_words (list[TextWord]): List of all TextWord objects.

        Returns:
            list[AToBSidebar]: List of all AToBSidebars identified.
        """
        entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=True)

        def find_pair(entry: DepthColumnEntry) -> DepthColumnEntry | None:  # noqa: D103
            min_y0 = entry.rect.y0 - entry.rect.height / 2
            max_y0 = entry.rect.y0 + entry.rect.height / 2
            for other in entries:
                if entry == other:
                    continue
                if other.value <= entry.value:
                    continue
                combined_width = entry.rect.width + other.rect.width
                if not entry.rect.x0 <= other.rect.x0 <= entry.rect.x0 + combined_width:
                    continue
                if not min_y0 <= other.rect.y0 <= max_y0:
                    continue
                in_between_text = " ".join(
                    [
                        word.text
                        for word in all_words
                        if entry.rect.x0 < word.rect.x0 < other.rect.x0 and min_y0 <= word.rect.y0 <= max_y0
                    ]
                )
                if re.fullmatch(r"\W*m?\W*", in_between_text):
                    return other

        pairs = [(entry, find_pair(entry)) for entry in entries]
        intervals = [AToBInterval(first, second) for first, second in pairs if second]
        clusters = Cluster[AToBInterval].create_clusters(intervals, lambda interval: interval.rect)
        merged_clusters = Cluster[AToBInterval].merge_close_clusters(clusters)
        clusters.extend(merged_clusters)

        detailed_sidebars = [AToBSidebar(get_most_detailed_intervals(clust.entries)) for clust in clusters]
        detailed_sidebars = [detailed_sidebar for detailed_sidebar in detailed_sidebars if detailed_sidebar.is_valid()]
        sidebars = [
            sidebar_segment
            for cluster in clusters
            for sidebar_segment in AToBSidebar(cluster.entries).break_on_mismatch()
            if sidebar_segment.is_valid()
        ]

        # return unique sidebars by comparing each intervals.
        return list({tuple(sb.entries): sb for sb in sidebars + detailed_sidebars}.values())

    def find_in_words_old(all_words: list[TextWord]) -> list[AToBSidebar]:
        """Finds all AToBSidebars.

        Generates a list of AToBDepthColumnEntry objects by finding consecutive pairs of DepthColumnEntry objects.
        Different columns are grouped together in LayerDepthColumn objects. Finally, a list of AToBSidebars objects,
        one for each column, is returned.

        A layer corresponds to a material layer. The layer is defined using a start and end point (e.g. 1.10-1.60m).
        The start and end points are represented as DepthColumnEntry objects.

        Args:
            all_words (list[TextWord]): List of all TextWord objects.

        Returns:
            list[AToBSidebar]: List of all AToBSidebars identified.
        """
        entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=True)

        def find_pair(entry: DepthColumnEntry) -> DepthColumnEntry | None:  # noqa: D103
            min_y0 = entry.rect.y0 - entry.rect.height / 2
            max_y0 = entry.rect.y0 + entry.rect.height / 2
            for other in entries:
                if entry == other:
                    continue
                if other.value <= entry.value:
                    continue
                combined_width = entry.rect.width + other.rect.width
                if not entry.rect.x0 <= other.rect.x0 <= entry.rect.x0 + combined_width:
                    continue
                if not min_y0 <= other.rect.y0 <= max_y0:
                    continue
                in_between_text = " ".join(
                    [
                        word.text
                        for word in all_words
                        if entry.rect.x0 < word.rect.x0 < other.rect.x0 and min_y0 <= word.rect.y0 <= max_y0
                    ]
                )
                if re.fullmatch(r"\W*m?\W*", in_between_text):
                    return other

        pairs = [(entry, find_pair(entry)) for entry in entries]
        intervals = [AToBInterval(first, second) for first, second in pairs if second]
        clusters = Cluster[AToBInterval].create_clusters(intervals, lambda interval: interval.rect)

        return [
            sidebar_segment
            for cluster in clusters
            for sidebar_segment in AToBSidebar(cluster.entries).break_on_mismatch()
            if sidebar_segment.is_valid()
        ]


def get_most_detailed_intervals(intervals: list[AToBInterval]) -> list[AToBInterval]:
    """Takes a list of intervals and returns the most detailed list of intervals.

    This is done by finding the list of intervals that can be continued from one to the other, that contains the most
    details and that reach the biggest depth.

    Args:
        intervals (list[AToBInterval]): The list of intervals.

    Returns:
        list[AToBInterval]: The ordered list
    """
    # Sort intervals by depth (interval close to the surface first) and thickness (small interval first).
    intervals.sort(key=lambda x: (x.start.value, x.end.value))

    current_intervals = [intervals[0]]
    processed_intervals = {intervals[0]}
    most_detailed_intervals = []  # stores the deepest list of continued interval found so far

    while current_intervals:
        current_interval = current_intervals[-1]
        found = False
        for interval in intervals:
            if interval in processed_intervals:
                continue
            # from the current (deepest) interval, look for interval that starts with the ending value (we stop at
            # the first interval found as they are sorted, so we will get the smallest one). We also ensure that the
            # interval found is below the current, to remove any noise.
            if current_interval.end.value == interval.start.value and current_interval.rect.y0 <= interval.rect.y0:
                current_intervals.append(interval)
                processed_intervals.add(interval)
                found = True
                break
        if not found:
            # if we did not found an interval to continue the search, we store the result if it is the best so far,
            # and continue the search from the previous interval, by poping the last added.
            if not most_detailed_intervals or current_intervals[-1].end.value > most_detailed_intervals[-1].end.value:
                most_detailed_intervals = current_intervals.copy()
            current_intervals.pop()
    return most_detailed_intervals
