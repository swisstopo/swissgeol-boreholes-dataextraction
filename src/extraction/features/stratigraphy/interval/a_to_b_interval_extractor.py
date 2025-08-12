"""Contains logic for finding AToBInterval instances in a text."""

import re

import pymupdf

from extraction.features.utils.text.textline import TextLine

from ...utils.text.textblock import TextBlock
from ..base.sidebar_entry import DepthColumnEntry
from .interval import AToBInterval, IntervalBlockPair


class AToBIntervalExtractor:
    """Methods for finding AToBInterval instances (e.g. "0.5m - 1.8m") in a text."""

    @staticmethod
    def from_material_description_lines(lines: list[TextLine]) -> list[IntervalBlockPair]:
        """Extract depth interval from text lines from a material description.

        For borehole profiles using the Deriaz layout, depth intervals are typically embedded within the material
        description text. These descriptions often further subdivide into multiple sublayers, each with its own
        distinct depth interval. This function extracts all such depth intervals found in the description, along with
        their corresponding text blocks. Decisions about which intervals to keep or discard are handled by downstream
        processing.
        For example (from GeoQuat 12306):
            1) REMBLAIS HETEROGENES
               0.00 - 0.08 m : Revêtement bitumineux
               0.08- 0.30 m : Grave d'infrastructure
               0.30 - 1.40 m : Grave dans importante matrice de sable
                               moyen, brun beige, pulvérulent.
        From this material description, this method will extract all depth intervals.

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
        page_number = text_line.page_number

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
                DepthColumnEntry.from_string_value(rect_from_group_index(1), match.group(1), page_number),
                DepthColumnEntry.from_string_value(rect_from_group_index(2), match.group(2), page_number),
            )
        return None
