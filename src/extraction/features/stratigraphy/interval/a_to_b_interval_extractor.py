"""Contains logic for finding AToBInterval instances in a text."""

import re

import pymupdf
from extraction.features.utils.text.textline import TextLine

from ...utils.text.textblock import TextBlock
from ..base.sidebar_entry import DepthColumnEntry
from .interval import AToBInterval, Interval, IntervalBlockPair


class AToBIntervalExtractor:
    """Methods for finding AToBInterval instances (e.g. "0.5m - 1.8m") in a text."""

    @staticmethod
    def from_material_description_lines(lines: list[TextLine]) -> list[IntervalBlockPair]:
        """Extract depth interval from text lines from a material description.

        TODO update docs

        For borehole profiles in the Deriaz layout, the depth interval is usually found in the text of the material
        description. Often, these text descriptions contain a further separation into multiple sub layers.
        These sub layers have their own depth intervals. This function extracts the overall depth interval,
        spanning across all mentioned sub layers.

        For example (from GeoQuat 12306):
            1) REMBLAIS HETEROGENES
               0.00 - 0.08 m : Revêtement bitumineux
               0.08- 0.30 m : Grave d'infrastructure
               0.30 - 1.40 m : Grave dans importante matrice de sable
                               moyen, brun beige, pulvérulent.
        From this material description, this method will extract a single depth interval that starts at 0m and ends
        at 1.40m.

        Args:
            lines (list[TextLine]): The lines to extract the depth interval from.

        Returns:
            list[IntervalBlockPair]: a list of interval-block-pairs that can be extracted from the given lines
        """
        entries = []
        current_block = []
        current_interval = None
        start_depth = None
        for line in lines:
            a_to_b_interval = AToBIntervalExtractor.from_text(line, require_start_of_string=False)
            # require_start_of_string = False because the depth interval may not always start at the beginning
            # of the line e.g. "Remblais Heterogene: 0.00 - 0.5m"
            if a_to_b_interval:
                # We assume that the first depth encountered is the start depth, and we reject further depth values
                # smaller than this first one. This avoids some false positives (e.g. GeoQuat 3339.pdf).
                if not start_depth:
                    start_depth = a_to_b_interval.start.value
                if a_to_b_interval.start.value >= start_depth:
                    if current_interval:
                        entries.append(IntervalBlockPair(current_interval, TextBlock(current_block)))
                        current_block = []
                    current_interval = a_to_b_interval
            current_block.append(line)
        entries.append(IntervalBlockPair(current_interval, TextBlock(current_block)))

        return entries

    @classmethod
    def from_text(cls, text_line: TextLine, require_start_of_string: bool = True) -> AToBInterval | None:
        """Attempts to extract a AToBInterval from a string.

        Args:
            text_line (TextLine): The text line to extract the depth interval from.
            require_start_of_string (bool, optional): Whether the number to extract needs to be
                                                      at the start of a string. Defaults to True.

        Returns:
            AToBInterval | None: The extracted AToBInterval or None if none is found.
        """
        input_string = text_line.text.strip().replace(",", ".")

        # for every character in input_string, list the index of the word this character originates from
        char_index_to_word_index = []
        for index, word in enumerate(text_line.words):
            char_index_to_word_index.extend([index] * (len(word.text) + 1))  # +1 to include the space between words

        query = r"-?([0-9]+(?:\.[0-9]+)?)[müMN\]*[\s-]+([0-9]+(?:\.[0-9]+)?)[müMN\\.]*"
        if not require_start_of_string:
            query = r".*?" + query
        regex = re.compile(query)
        match = regex.match(input_string)

        def rect_from_group_index(index):
            """Give the rect that covers all the words that intersect with the given regex group."""
            rect = pymupdf.Rect()
            start_word_index = char_index_to_word_index[match.start(index)]
            # `match.end(index) - 1`, because match.end gives the index of the first character that is *not* matched,
            # whereas we want the last character that *is* matched.
            end_word_index = char_index_to_word_index[match.end(index) - 1]
            # `end_word_index + 1` because the end of the range is exclusive by default, whereas we also want to
            # include the word with this index
            for word_index in range(start_word_index, end_word_index + 1):
                rect.include_rect(text_line.words[word_index].rect)
            return rect

        if match:
            return AToBInterval(
                DepthColumnEntry.from_string_value(rect_from_group_index(1), match.group(1)),
                DepthColumnEntry.from_string_value(rect_from_group_index(2), match.group(2)),
            )
        return None

    @staticmethod
    def number_of_subintervals(interval: Interval, following_intervals: list[Interval]) -> bool:
        count = 0
        for following_interval in following_intervals:
            if interval.start.value <= following_interval.start.value <= interval.end.value and (
                interval.start.value <= following_interval.end.value <= interval.end.value
            ):
                count += 1
            else:
                break
        return count

    @staticmethod
    def is_partitioned(interval: Interval, following_intervals: list[Interval]) -> bool:
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

    @staticmethod
    def partitions_and_sublayers(intervals: list[Interval]) -> list[Interval]:
        """Takes a list of intervals and returns the most detailed list of intervals.

        Args:
            intervals (list[Interval]): The list of intervals.

        Returns:
            list[Interval]: The ordered list
        """
        intervals = intervals.copy()  # don't mutate the original object

        continue_search = True
        while continue_search:
            continue_search = False
            for index, interval in enumerate(intervals):
                if not interval.skip_interval:
                    following_intervals = [
                        interval for interval in intervals[index + 1 :] if not interval.skip_interval
                    ]
                    if AToBIntervalExtractor.is_partitioned(interval, following_intervals):
                        intervals[index].is_parent = True
                        continue_search = True
                        break

        continue_search = True
        while continue_search:
            continue_search = False
            filtered_intervals = [interval for interval in intervals if not interval.skip_interval]
            for index, interval in enumerate(filtered_intervals):
                subinterval_count = AToBIntervalExtractor.number_of_subintervals(
                    interval, filtered_intervals[index + 1 :]
                )
                if subinterval_count > 0:
                    for step in range(subinterval_count):
                        filtered_intervals[index + step + 1].is_sublayer = True
                    continue_search = True
                    break

        return intervals

    @staticmethod
    def partitions_and_sublayers_with_text(pairs: list[IntervalBlockPair]) -> list[IntervalBlockPair]:
        pairs = pairs.copy()  # don't mutate the original object
        intervals = [pair.depth_interval for pair in pairs if pair.depth_interval]
        AToBIntervalExtractor.partitions_and_sublayers(intervals)

        processed_pairs = []
        current_block = None
        current_interval = None
        for pair in pairs:
            if pair.depth_interval and pair.depth_interval.skip_interval:
                current_block = current_block.concatenate(pair.block) if current_block else pair.block
            else:
                if current_interval:
                    processed_pairs.append(IntervalBlockPair(current_interval, current_block))
                    current_block = pair.block
                else:
                    current_block = current_block.concatenate(pair.block) if current_block else pair.block
                current_interval = pair.depth_interval

        processed_pairs.append(IntervalBlockPair(current_interval, current_block))
        return processed_pairs
