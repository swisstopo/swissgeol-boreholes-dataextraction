"""Contains logic for finding AToBInterval instances in a text."""

import re

import pymupdf

from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

from ...utils.text.textblock import TextBlock
from ..base.sidebar_entry import DepthColumnEntry
from .interval import AToBInterval, IntervalBlockPair

matching_params = read_params("matching_params.yml")


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
        prev_line = None
        prev_interval = None
        for idx, line in enumerate(lines):
            a_to_b_interval, line_without_depths = AToBIntervalExtractor.from_text(line, require_start_of_string=False)
            # First line of the block is stripped of its (potential) leading depths
            final_line = line if idx != 0 else line_without_depths
            if prev_line and not a_to_b_interval and not prev_interval:
                # if depth was not found in the previous and current lines, we look for a depth wrapping arround.
                combined_lines = TextLine(prev_line.words + line.words)
                a_to_b_interval, _ = AToBIntervalExtractor.from_text(combined_lines, require_start_of_string=False)
            prev_interval = a_to_b_interval
            prev_line = line
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
                        final_line = line_without_depths  # start of a new depth block, strip leading depth
                    current_interval = a_to_b_interval
            if final_line and final_line.words:
                current_block.append(final_line)
        if current_block:
            entries.append(IntervalBlockPair(current_interval, TextBlock(current_block)))

        return entries

    @classmethod
    def from_text(
        cls, text_line: TextLine, require_start_of_string: bool = True
    ) -> tuple[AToBInterval | None, TextLine | None]:
        """Attempts to extract a AToBInterval from a string.

        Args:
            text_line (TextLine): The text line to extract the depth interval from.
            require_start_of_string (bool, optional): Whether the number to extract needs to be
                                                      at the start of a string. Defaults to True.

        Returns:
            tuple[AToBInterval | None, TextLine | None]: The extracted AToBInterval and the TextLine without the
                depth related words or None if none is found.
        """
        input_string = text_line.text

        # for every character in input_string, list the index of the word this character originates from
        char_index_to_word_index = []
        for index, word in enumerate(text_line.words):
            char_index_to_word_index.extend([index] * (len(word.text) + 1))  # +1 to include the space between words

        number_capturing = r"([0-9]+(?:[\.,][0-9]+)?)"
        unit = r"(?:[müMN\s]*(?![A-Za-z]))?"

        query = (
            rf"(-?{number_capturing}{unit}"
            r"[\s-]+"
            rf"{number_capturing}{unit}"
            r"[\s\\.:;]*)"
        )

        if not require_start_of_string:
            query = r".*?" + query
        regex = re.compile(query)
        depths_match = regex.match(input_string)

        def rect_from_group_index(index):
            """Give the rect that covers all the words that intersect with the given regex group."""
            rect = pymupdf.Rect()
            start_word_index = char_index_to_word_index[depths_match.start(index)]
            # `match.end(index) - 1`, because match.end gives the index of the first character that is *not* matched,
            # whereas we want the last character that *is* matched.
            end_word_index = char_index_to_word_index[depths_match.end(index) - 1]
            # `end_word_index + 1` because the end of the range is exclusive by default, whereas we also want to
            # include the word with this index
            for word_index in range(start_word_index, end_word_index + 1):
                rect.include_rect(text_line.words[word_index].rect)
            return rect

        def remaining_line() -> TextLine:
            if char_index_to_word_index[depths_match.start(1)] != 0:  # group 1 is the whole depth matching
                return text_line  # the depths found do not start the line
            return TextLine(
                [w for i, w in enumerate(text_line.words) if i > char_index_to_word_index[depths_match.end(1) - 1]]
            )

        if depths_match:
            return (
                AToBInterval(
                    DepthColumnEntry.from_string_value(rect_from_group_index(2), depths_match.group(2)),
                    DepthColumnEntry.from_string_value(rect_from_group_index(3), depths_match.group(3)),
                ),
                remaining_line(),
            )

        open_ended_words = matching_params["open_ended_depth_key"]
        words_pattern = "|".join([re.escape(w) for w in open_ended_words])

        fallback_query = rf"((?:{words_pattern})\s*([0-9]+(?:[\.,][0-9]+)?)\s*[müMN]*)"
        if not require_start_of_string:
            fallback_query = r".*?" + fallback_query
        fallback_regex = re.compile(fallback_query, re.IGNORECASE)
        depths_match = fallback_regex.search(input_string)
        if depths_match:
            return AToBInterval(
                DepthColumnEntry.from_string_value(rect_from_group_index(2), depths_match.group(2)), None
            ), remaining_line()

        return None, text_line
