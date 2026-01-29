"""Contains logic for finding AToBInterval instances in a text."""

import re

import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.interval import AToBInterval
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params

matching_params = read_params("matching_params.yml")


class AToBIntervalExtractor:
    """Methods for finding AToBInterval instances (e.g. "0.5m - 1.8m") in a text."""

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
        page_number = text_line.page_number

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
            return TextLine(text_line.words[char_index_to_word_index[depths_match.end(1) - 1] + 1 :])

        if depths_match:
            return (
                AToBInterval(
                    DepthColumnEntry.from_string_value(rect_from_group_index(2), depths_match.group(2), page_number),
                    DepthColumnEntry.from_string_value(rect_from_group_index(3), depths_match.group(3), page_number),
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
                DepthColumnEntry.from_string_value(rect_from_group_index(2), depths_match.group(2), page_number), None
            ), remaining_line()

        return None, text_line
