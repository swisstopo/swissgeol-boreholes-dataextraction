"""Contains logic for finding depth column entries in text."""

import re

from extraction.features.stratigraphy.sidebarentry.depth_column_entry import DepthColumnEntry
from swissgeol_doc_processing.text.textline import TextWord


class DepthColumnEntryExtractor:
    """Methods for finding depth column entries in a text."""

    regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]*)?)m?\.?$")

    @classmethod
    def find_in_word(cls, word: TextWord) -> DepthColumnEntry | None:
        """Find a depth column entry in a single TextWord, if there is one.

        Note: Only depths up to two digits before the decimal point are supported.

        Args:
            word (TextWord): The word to extract a depth column entry from.

        Returns:
            DepthColumnEntry | None: The extracted depth column entry, or None if the word is not a depth value.
        """
        try:
            input_string = word.text.strip().replace(",", ".")
            # numbers such as '.40' are not supported. The reason is that sometimes the OCR
            # recognizes a '-' as a '.' and we just ommit the leading '.' to avoid this issue.
            match = cls.regex.match(input_string)
            if match:
                return DepthColumnEntry.from_string_value(word.rect, match.group(1), word.page_number)
        except ValueError:
            pass
        return None

    @classmethod
    def find_in_words(cls, all_words: list[TextWord]) -> list[DepthColumnEntry]:
        """Find all depth column entries given a list of TextWord objects.

        Note: Only depths up to two digits before the decimal point are supported.

        Args:
            all_words (list[TextWord]): List of text words to extract depth column entries from.

        Returns:
            list[DepthColumnEntry]: The extracted depth column entries.
        """
        entries = []
        for word in sorted(all_words, key=lambda word: word.rect.y0):
            if entry := cls.find_in_word(word):
                entries.append(entry)
        return entries
