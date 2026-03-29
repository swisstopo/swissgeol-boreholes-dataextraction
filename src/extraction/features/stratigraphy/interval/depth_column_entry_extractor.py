"""Contains logic for finding depth column entries in text."""

import re

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from swissgeol_doc_processing.text.textline import TextWord


class DepthColumnEntryExtractor:
    """Methods for finding depth column entries in a text."""

    @classmethod
    def find_in_words(cls, all_words: list[TextWord]) -> list[DepthColumnEntry]:
        """Find all depth column entries given a list of TextWord objects.

        Note: Only depths up to two digits before the decimal point are supported.

        Args:
            all_words (list[TextWord]): List of text words to extract depth column entries from.
            include_splits (bool): Whether to include split entries.

        Returns:
            list[DepthColumnEntry]: The extracted depth column entries.
        """
        entries = []
        regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]*)?)[müMN\\.]*$")

        for word in sorted(all_words, key=lambda word: word.rect.y0):
            try:
                input_string = word.text.strip().replace(",", ".")
                # numbers such as '.40' are not supported. The reason is that sometimes the OCR
                # recognizes a '-' as a '.' and we just ommit the leading '.' to avoid this issue.
                match = regex.match(input_string)
                if match:
                    entries.append(DepthColumnEntry.from_string_value(word.rect, match.group(1), word.page_number))
            except ValueError:
                pass
        return entries
