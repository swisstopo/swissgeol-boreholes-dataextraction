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
        return [
            sidebar_segment
            for cluster in clusters
            for sidebar_segment in AToBSidebar(get_most_detailed_intervals(cluster.entries)).break_on_mismatch()
            if sidebar_segment.is_valid()
        ]


def _is_partitioned(interval: AToBInterval, following_intervals: list[AToBInterval]) -> bool:
    current_end = interval.start.value
    for following_interval in following_intervals:
        if following_interval.start.value == current_end:
            current_end = following_interval.end.value
            if current_end == interval.end.value:
                return True
            if current_end > interval.end.value:
                return False
        else:
            return False
    return False


def _number_of_subintervals(interval: AToBInterval, following_intervals: list[AToBInterval]) -> bool:
    count = 0
    for following_interval in following_intervals:
        if interval.start.value <= following_interval.start.value <= interval.end.value and (
            interval.start.value <= following_interval.end.value <= interval.end.value
        ):
            count += 1
        else:
            break
    return count


def get_most_detailed_intervals(intervals: list[AToBInterval]) -> list[AToBInterval]:
    """Takes a list of intervals and returns the most detailed list of intervals.

    This is done by finding the list of intervals that can be continued from one to the other, that contains the most
    details and that reach the biggest depth.

    Args:
        intervals (list[AToBInterval]): The list of intervals.

    Returns:
        list[AToBInterval]: The ordered list
    """
    intervals = intervals.copy()  # don't mutate the original object

    continue_search = True
    while continue_search:
        continue_search = False
        for index, interval in enumerate(intervals):
            if _is_partitioned(interval, intervals[index + 1 :]):
                # TODO: instead of removing this interval, keep it, but mark it as a "parent interval", so that the
                # corresponding description does not get appended to the preceding interval.
                intervals.pop(index)
                continue_search = True
                break

    continue_search = True
    while continue_search:
        continue_search = False
        for index, interval in enumerate(intervals):
            subinterval_count = _number_of_subintervals(interval, intervals[index + 1 :])
            if subinterval_count > 0:
                del intervals[index + 1 : index + subinterval_count + 1]
                continue_search = True
                break

    return intervals
