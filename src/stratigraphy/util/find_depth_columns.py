"""This module contains functionalities to find depth columns in a pdf page."""

import re

import fitz

from stratigraphy.util.depthcolumn import BoundaryDepthColumn, LayerDepthColumn
from stratigraphy.util.depthcolumnentry import DepthColumnEntry, LayerDepthColumnEntry
from stratigraphy.util.line import TextLine


def depth_column_entries(all_words: list[TextLine], include_splits: bool) -> list[DepthColumnEntry]:
    """Find all depth column entries given a list of TextLine objects.

    Args:
        all_words (list[TextLine]): List of Text lines to extract depth column entries from.
        include_splits (bool): Whether to include split entries.

    Returns:
        list[DepthColumnEntry]: The extracted depth column entries.
    """

    def value_as_float(string_value: str) -> float:  # noqa: D103
        # OCR sometimes tends to miss the decimal comma
        parsed_text = re.sub(r"^([0-9]+)([0-9]{2})", r"\1.\2", string_value)
        return abs(float(parsed_text))

    entries = []
    for line in sorted(all_words, key=lambda line: line.rect.y0):
        try:
            input_string = line.text.strip().replace(",", ".")
            regex = re.compile(r"^([0-9]+(\.[0-9]+)?)[müMN\\.]*$")
            match = regex.match(input_string)

            if match:
                value = value_as_float(match.group(1))
                entries.append(DepthColumnEntry(line.rect, value))
            elif include_splits:
                # support for e.g. "1.10-1.60m" extracted as a single word
                regex2 = re.compile(r"^([0-9]+(\.[0-9]+)?)[müMN\\.]*\W+([0-9]+(\.[0-9]+)?)[müMN\\.]*$")
                match2 = regex2.match(input_string)

                if match2:
                    value1 = value_as_float(match2.group(1))
                    first_half_rect = fitz.Rect(
                        line.rect.x0, line.rect.y0, line.rect.x1 - line.rect.width / 2, line.rect.y1
                    )
                    entries.append(DepthColumnEntry(first_half_rect, value1))

                    value2 = value_as_float(match2.group(3))
                    second_half_rect = fitz.Rect(
                        line.rect.x0 + line.rect.width / 2, line.rect.y0, line.rect.x1, line.rect.y1
                    )
                    entries.append(DepthColumnEntry(second_half_rect, value2))
        except ValueError:
            pass
    return entries


def find_layer_depth_columns(entries: list[DepthColumnEntry], all_words: list[TextLine]) -> list[LayerDepthColumn]:
    """TODO: Add description here. It is not entirely clear to me (@redur) what this function does.

    Args:
        entries (list[DepthColumnEntry]): _description_
        all_words (list[TextLine]): _description_

    Returns:
        list[LayerDepthColumn]: _description_
    """

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

    columns = []
    for first, second in pairs:
        if second is not None:
            entry = LayerDepthColumnEntry(first, second)
            is_matched = False
            for column in columns:
                column_rect = column.rect()
                new_start_middle = (entry.start.rect.x0 + entry.start.rect.x1) / 2
                if column_rect.x0 < new_start_middle < column_rect.x1:
                    is_matched = True
                    column.add_entry(entry)

            if not is_matched:
                columns.append(LayerDepthColumn([entry]))

    return [
        column_segment
        for column in columns
        for column_segment in column.break_on_mismatch()
        if column_segment.is_valid()
    ]


def find_depth_columns(entries: list[DepthColumnEntry], all_words: list[TextLine]) -> list[BoundaryDepthColumn]:
    """TODO: Add description here. It is not entirely clear to me (@redur) what this function does.

    Args:
        entries (list[DepthColumnEntry]): _description_
        all_words (list[TextLine]): _description_

    Returns:
        list[BoundaryDepthColumn]: _description_
    """
    numeric_columns: list[BoundaryDepthColumn] = []
    for entry in entries:
        has_match = False
        additional_columns = []
        for column in numeric_columns:
            if column.can_be_appended(entry.rect):
                has_match = True
                column.add_entry(entry)
            else:
                valid_initial_segment = column.valid_initial_segment(entry.rect)
                if len(valid_initial_segment.entries) > 0:
                    has_match = True
                    additional_columns.append(valid_initial_segment.add_entry(entry))

        numeric_columns.extend(additional_columns)
        if not has_match:
            numeric_columns.append(BoundaryDepthColumn([entry]))

        # only keep columns that are not contained in a different column
        numeric_columns = [
            column
            for column in numeric_columns
            if all(not other.strictly_contains(column) for other in numeric_columns)
        ]

    numeric_columns = [
        column.reduce_until_valid(all_words)
        for numeric_column in numeric_columns
        for column in numeric_column.break_on_double_descending()
        # when we have a perfect arithmetic progression, this is usually just a scale
        # that does not match the descriptions
        if not column.significant_arithmetic_progression()
    ]

    return sorted(
        [column for column in numeric_columns if column and column.is_valid(all_words)],
        key=lambda column: len(column.entries),
    )
