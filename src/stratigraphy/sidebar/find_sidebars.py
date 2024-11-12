"""This module contains functionalities to find sidebars in a pdf page."""

import re

import fitz

from stratigraphy.depthcolumn.depthcolumnentry import AToBDepthColumnEntry, DepthColumnEntry
from stratigraphy.lines.line import TextWord
from stratigraphy.sidebar import AAboveBSidebar, AAboveBSidebarValidator, AToBSidebar
from stratigraphy.text.textblock import TextBlock
from stratigraphy.util.interval import AToBInterval


def depth_column_entries(all_words: list[TextWord], include_splits: bool) -> list[DepthColumnEntry]:
    """Find all depth column entries given a list of TextLine objects.

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
                layer_depth_column_entry = extract_layer_depth_interval(input_string, word.rect)
                entries.extend(
                    [layer_depth_column_entry.start, layer_depth_column_entry.end] if layer_depth_column_entry else []
                )
        except ValueError:
            pass
    return entries


def value_as_float(string_value: str) -> float:  # noqa: D103
    """Converts a string to a float."""
    # OCR sometimes tends to miss the decimal comma
    parsed_text = re.sub(r"^-?([0-9]+)([0-9]{2})", r"\1.\2", string_value)
    return abs(float(parsed_text))


def extract_layer_depth_interval(
    text: str, rect: fitz.Rect, require_start_of_string: bool = True
) -> AToBDepthColumnEntry | None:
    """Extracts a LayerDepthColumnEntry from a string.

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


def find_a_to_b_sidebars(entries: list[DepthColumnEntry], all_words: list[TextWord]) -> list[AToBSidebar]:
    """Finds all AToBSidebars.

    Generates a list of LayerDepthColumnEntry objects by finding consecutive pairs of DepthColumnEntry objects.
    Different columns are grouped together in LayerDepthColumn objects. Finally, a list of AToBSidebars objects,
    one for each column, is returned.

    A layer corresponds to a material layer. The layer is defined using a start and end point (e.g. 1.10-1.60m).
    The start and end points are represented as DepthColumnEntry objects.

    Args:
        entries (list[DepthColumnEntry]): List of depth column entries.
        all_words (list[TextWord]): List of all TextWord objects.

    Returns:
        list[AToBSidebar]: List of all AToBSidebars identified.
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

    sidebars = []
    for first, second in pairs:
        if second is not None:
            entry = AToBDepthColumnEntry(first, second)
            is_matched = False
            for sidebar in sidebars:
                column_rect = sidebar.rect()
                new_start_middle = (entry.start.rect.x0 + entry.start.rect.x1) / 2
                if column_rect.x0 < new_start_middle < column_rect.x1:
                    is_matched = True
                    sidebar.entries.append(entry)

            if not is_matched:
                sidebars.append(AToBSidebar([entry]))

    return [
        sidebar_segment
        for sidebar in sidebars
        for sidebar_segment in sidebar.break_on_mismatch()
        if sidebar_segment.is_valid()
    ]


def find_a_above_b_sidebars(
    entries: list[DepthColumnEntry], all_words: list[TextWord], sidebar_params: dict
) -> list[AAboveBSidebar]:
    """Construct all possible AAboveBSidebar objects from the given DepthColumnEntry objects.

    Args:
        entries (list[DepthColumnEntry]): All found depth column entries in the page.
        all_words (list[TextLine]): All words in the page.
        sidebar_params (dict): Parameters for the BoundaryDepthColumn objects.

    Returns:
        list[AAboveBSidebar]: Found BoundaryDepthColumn objects.
    """
    numeric_columns: list[AAboveBSidebar] = []
    for entry in entries:
        has_match = False
        additional_columns = []
        for column in numeric_columns:
            if column.can_be_appended(entry.rect):
                has_match = True
                column.entries.append(entry)
            else:
                valid_initial_segment = column.valid_initial_segment(entry.rect)
                if len(valid_initial_segment.entries) > 0:
                    has_match = True
                    valid_initial_segment.entries.append(entry)
                    additional_columns.append(valid_initial_segment)

        numeric_columns.extend(additional_columns)
        if not has_match:
            numeric_columns.append(AAboveBSidebar(entries=[entry]))

        # only keep columns that are not contained in a different column
        numeric_columns = [
            column
            for column in numeric_columns
            if all(not other.strictly_contains(column) for other in numeric_columns)
        ]

    boundary_depth_column_validator = AAboveBSidebarValidator(all_words, **sidebar_params)

    numeric_columns = [
        boundary_depth_column_validator.reduce_until_valid(column)
        for numeric_column in numeric_columns
        for column in numeric_column.break_on_double_descending()
        # when we have a perfect arithmetic progression, this is usually just a scale
        # that does not match the descriptions
        if not column.significant_arithmetic_progression()
    ]

    return sorted(
        [column for column in numeric_columns if column and boundary_depth_column_validator.is_valid(column)],
        key=lambda column: len(column.entries),
    )


def get_depth_interval_from_textblock(block: TextBlock) -> AToBInterval | None:
    """Extract depth interval from a material description block.

    For borehole profiles in the Deriaz layout, the depth interval is usually found in the text description
    of the material. Often, these text descriptions contain a further separation into multiple sub layers.
    These sub layers have their own depth intervals. This function extracts the overall depth interval,
    spanning across all mentioned sub layers.

    Args:
        block (TextBlock): The block to calculate the depth interval for.

    Returns:
        AToBInterval | None: The depth interval.
    """
    depth_entries = []
    for line in block.lines:
        try:
            layer_depth_entry = extract_layer_depth_interval(line.text, line.rect, require_start_of_string=False)
            # require_start_of_string = False because the depth interval may not always start at the beginning
            # of the line e.g. "Remblais Heterogene: 0.00 - 0.5m"
            if layer_depth_entry:
                depth_entries.append(layer_depth_entry)
        except ValueError:
            pass

    if depth_entries:
        # Merge the sub layers into one depth interval.
        start = min([entry.start for entry in depth_entries], key=lambda start_entry: start_entry.value)
        end = max([entry.end for entry in depth_entries], key=lambda end_entry: end_entry.value)

        return AToBInterval(AToBDepthColumnEntry(start, end))
    else:
        return None
