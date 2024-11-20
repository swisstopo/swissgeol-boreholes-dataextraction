"""Contains logic for finding AToBInterval instances in a text."""

import re

import fitz

from stratigraphy.lines.line import TextLine

from .depthcolumnentry import DepthColumnEntry
from .interval import AToBInterval
from .util import value_as_float


class AToBIntervalExtractor:
    """Methods for finding AToBInterval instances (e.g. "0.5m - 1.8m") in a text."""

    @classmethod
    def from_lines(cls, lines: list[TextLine]) -> AToBInterval | None:
        """Extract depth interval from text lines.

        For borehole profiles in the Deriaz layout, the depth interval is usually found in the text of the material
        description. Often, these text descriptions contain a further separation into multiple sub layers.
        These sub layers have their own depth intervals. This function extracts the overall depth interval,
        spanning across all mentioned sub layers.

        Args:
            lines (list[TextLine]): The lines to extract the depth interval from.

        Returns:
            AToBInterval | None: The depth interval (if any) or None (if no depth interval was found).
        """
        depth_entries = []
        for line in lines:
            try:
                a_to_b_depth_entry = AToBIntervalExtractor.from_text(
                    line.text, line.rect, require_start_of_string=False
                )
                # require_start_of_string = False because the depth interval may not always start at the beginning
                # of the line e.g. "Remblais Heterogene: 0.00 - 0.5m"
                if a_to_b_depth_entry:
                    depth_entries.append(a_to_b_depth_entry)
            except ValueError:
                pass

        if depth_entries:
            # Merge the sub layers into one depth interval.
            start = min([entry.start for entry in depth_entries], key=lambda start_entry: start_entry.value)
            end = max([entry.end for entry in depth_entries], key=lambda end_entry: end_entry.value)
            return AToBInterval(start, end)
        else:
            return None

    @classmethod
    def from_text(cls, text: str, rect: fitz.Rect, require_start_of_string: bool = True) -> AToBInterval | None:
        """Attempts to extract a AToBInterval from a string.

        Args:
            text (str): The string to extract the depth interval from.
            rect (fitz.Rect): The rectangle of the text.
            require_start_of_string (bool, optional): Whether the number to extract needs to be
                                                      at the start of a string. Defaults to True.

        Returns:
            AToBInterval | None: The extracted AToBInterval or None if none is found.
        """
        input_string = text.strip().replace(",", ".")

        query = r"-?([0-9]+(\.[0-9]+)?)[müMN\]*[\s-]+([0-9]+(\.[0-9]+)?)[müMN\\.]*"
        if not require_start_of_string:
            query = r".*?" + query
        regex = re.compile(query)
        match = regex.match(input_string)
        if match:
            value1 = value_as_float(match.group(1))
            first_half_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 - rect.width / 2, rect.y1)

            value2 = value_as_float(match.group(3))
            second_half_rect = fitz.Rect(rect.x0 + rect.width / 2, rect.y0, rect.x1, rect.y1)
            return AToBInterval(
                DepthColumnEntry(first_half_rect, value1),
                DepthColumnEntry(second_half_rect, value2),
            )
        return None
