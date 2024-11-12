"""Contains dataclasses for entries in a depth column."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz
from stratigraphy.lines.line import TextWord


@dataclass
class DepthColumnEntry:  # noqa: D101
    """Class to represent a depth column entry."""

    rect: fitz.Rect
    value: float

    def __repr__(self) -> str:
        return str(self.value)

    def to_json(self) -> dict[str, Any]:
        """Convert the depth column entry to a JSON serializable format."""
        return {"value": self.value, "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1]}

    @classmethod
    def from_json(cls, data: dict) -> DepthColumnEntry:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the depth column entry.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(rect=fitz.Rect(data["rect"]), value=data["value"])

    @classmethod
    def find_in_words(cls, all_words: list[TextWord], include_splits: bool) -> list[DepthColumnEntry]:
        """Find all depth column entries given a list of TextWord objects.

        Note: Only depths up to two digits before the decimal point are supported.

        Args:
            all_words (list[TextWord]): List of text words to extract depth column entries from.
            include_splits (bool): Whether to include split entries.

        Returns:
            list[DepthColumnEntry]: The extracted depth column entries.
        """
        entries = []
        for word in sorted(all_words, key=lambda word: word.rect.y0):
            try:
                input_string = word.text.strip().replace(",", ".")
                regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]+)?)[müMN\\.]*$")
                # numbers such as '.40' are not supported. The reason is that sometimes the OCR
                # recognizes a '-' as a '.' and we just ommit the leading '.' to avoid this issue.
                match = regex.match(input_string)
                if match:
                    value = value_as_float(match.group(1))
                    entries.append(DepthColumnEntry(word.rect, value))
                elif include_splits:
                    # support for e.g. "1.10-1.60m" extracted as a single word
                    a_to_b_depth_column_entry = AToBDepthColumnEntry.from_text(input_string, word.rect)
                    entries.extend(
                        [a_to_b_depth_column_entry.start, a_to_b_depth_column_entry.end]
                        if a_to_b_depth_column_entry
                        else []
                    )
            except ValueError:
                pass
        return entries


@dataclass
class AToBDepthColumnEntry:  # noqa: D101
    """Class to represent a depth column entry of the form "1m - 3m"."""

    # TODO do we need both this class as well as AToBInterval, or can we combine the two classes?

    start: DepthColumnEntry
    end: DepthColumnEntry

    def __repr__(self) -> str:
        return f"{self.start.value}-{self.end.value}"

    @property
    def rect(self) -> fitz.Rect:
        """Get the rectangle of the layer depth column entry."""
        return fitz.Rect(self.start.rect).include_rect(self.end.rect)

    def to_json(self) -> dict[str, Any]:
        """Convert the layer depth column entry to a JSON serializable format."""
        return {"start": self.start.to_json(), "end": self.end.to_json()}

    @classmethod
    def from_json(cls, data: dict) -> AToBDepthColumnEntry:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depth column entry.

        Returns:
            AToBDepthColumnEntry: The A-to-B depth column entry object.
        """
        start = DepthColumnEntry.from_json(data["start"])
        end = DepthColumnEntry.from_json(data["end"])
        return cls(start, end)

    @classmethod
    def from_text(
        cls, text: str, rect: fitz.Rect, require_start_of_string: bool = True
    ) -> AToBDepthColumnEntry | None:
        """Attempts to extract a AToBDepthColumnEntry from a string.

        Args:
            text (str): The string to extract the depth interval from.
            rect (fitz.Rect): The rectangle of the text.
            require_start_of_string (bool, optional): Whether the number to extract needs to be
                                                      at the start of a string. Defaults to True.

        Returns:
            AToBDepthColumnEntry | None: The extracted LayerDepthColumnEntry or None if none is found.
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
            return AToBDepthColumnEntry(
                DepthColumnEntry(first_half_rect, value1),
                DepthColumnEntry(second_half_rect, value2),
            )
        return None


def value_as_float(string_value: str) -> float:  # noqa: D103
    """Converts a string to a float."""
    # OCR sometimes tends to miss the decimal comma
    parsed_text = re.sub(r"^-?([0-9]+)([0-9]{2})", r"\1.\2", string_value)
    return abs(float(parsed_text))
