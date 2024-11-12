"""Module for the layer identifier sidebars."""

import re
from dataclasses import dataclass

import fitz

from stratigraphy.layer.layer import IntervalBlockGroup
from stratigraphy.lines.line import TextLine
from stratigraphy.text.textblock import TextBlock
from stratigraphy.util.dataclasses import Line

from ..depthcolumn.depthcolumnentry import AToBDepthColumnEntry
from ..util.interval import AToBInterval
from .sidebar import Sidebar


class LayerIdentifierEntry:
    """Class for a layer identifier entry.

    Note: As of now this is very similar to DepthColumnEntry. Refactoring may be desired.
    """

    def __init__(self, rect: fitz.Rect, text: str):
        self.rect = rect
        self.text = text

    def __repr__(self):
        return str(self.text)


@dataclass
class LayerIdentifierSidebar(Sidebar[LayerIdentifierEntry]):
    """Class for a layer identifier sidebar.

    Layer identifiers are labels that are particularly common in Deriaz layout borehole profiles. They can be
    sequential such as in 1007.pdf - a), b), c), etc. - or contain some semantic meaning such as in 10781.pdf -
    5c12), 4a), etc.
    """

    entries: list[LayerIdentifierEntry]

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: fitz.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Divide the description lines into blocks based on the layer identifier entries.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.
            params (dict): A dictionary of relevant parameters.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.
        """
        blocks = []
        line_index = 0
        for layer_identifier_idx, _layer_index in enumerate(self.entries):
            next_layer_identifier = (
                self.entries[layer_identifier_idx + 1] if layer_identifier_idx + 1 < len(self.entries) else None
            )

            matched_block = self.matching_blocks(description_lines, line_index, next_layer_identifier)
            line_index += sum([len(block.lines) for block in matched_block])
            blocks.extend(matched_block)

        result = []
        for block in blocks:
            depth_intervals = []
            depth_interval = get_depth_interval_from_textblock(block)
            if depth_interval:
                depth_intervals.append(depth_interval)
            result.append(IntervalBlockGroup(depth_intervals=depth_intervals, blocks=[block]))

        return result

    @staticmethod
    def matching_blocks(
        all_lines: list[TextLine], line_index: int, next_layer_identifier: LayerIdentifierEntry | None
    ) -> list[TextBlock]:
        """Adds lines to a block until the next layer identifier is reached.

        Args:
            all_lines (list[TextLine]): All TextLine objects constituting the material description.
            line_index (int): The index of the last line that is already assigned to a block.
            next_layer_identifier (TextLine | None): The next layer identifier.

        Returns:
            list[TextBlock]: The next block or an empty list if no lines are added.
        """
        y1_threshold = None
        if next_layer_identifier:
            next_interval_start_rect = next_layer_identifier.rect
            y1_threshold = next_interval_start_rect.y0 + next_interval_start_rect.height / 2

        matched_lines = []

        for current_line in all_lines[line_index:]:
            if y1_threshold is None or current_line.rect.y1 < y1_threshold:
                matched_lines.append(current_line)
            else:
                break

        if matched_lines:
            return [TextBlock(matched_lines)]
        else:
            return []

    def strictly_contains(self, other: "LayerIdentifierSidebar") -> bool:
        """Check if the layer identifier column strictly contains another layer identifier column.

        Args:
            other (LayerIdentifierSidebar): The other layer identifier column to check if it is strictly contained.

        Returns:
            bool: True if the layer identifier column strictly contains the other layer identifier column, False
            otherwise.
        """
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_contained(self, rect: fitz.Rect) -> bool:
        """Check if the layer identifier column is contained in another rectangle.

        Args:
            rect (fitz.Rect): The rectangle to check if it contains the layer identifier column.

        Returns:
            bool: True if the layer identifier column is contained in the rectangle, False otherwise.
        """
        return (
            rect.x0 <= self.rect().x0
            and self.rect().x1 <= rect.x1
            and rect.y0 <= self.rect().y0
            and self.rect().y1 <= rect.y1
        )


def find_layer_identifier_sidebar_entries(lines: list[TextLine]) -> list[LayerIdentifierEntry]:
    r"""Find the layer identifier sidebar entries.

    Regex explanation:
    - \b is a word boundary. This ensures that the match must start at the beginning of a word.
    - [\da-z]+ matches one or more (+) alphanumeric characters (\d for digits and a-z for lowercase letters).
    - \) matches a closing parenthesis. The backslash is necessary because parentheses are special characters
      in regular expressions, so we need to escape it to match a literal parenthesis.
    This regular expression will match strings like "1)", "2)", "a)", "b)", "1a4)", "6de)", etc.

    Args:
        lines (list[TextLine]): The lines to search for layer identifier entries.

    Returns:
        list[LayerIdentifierEntry]: The layer identifier sidebar entries.
    """
    entries = []
    for line in sorted(lines, key=lambda line: line.rect.y0):
        if len(line.words) > 0:
            # Only match in the first word of every line, to avoid e.g. matching with "cm)" in a material description
            # containing an expression like "(diameter max 6 cm)".
            first_word = line.words[0]
            regex = re.compile(r"\b[\da-z-]+\)")
            match = regex.match(first_word.text)
            if match and len(first_word.text) < 7:
                entries.append(LayerIdentifierEntry(first_word.rect, first_word.text))
    return entries


def find_layer_identifier_sidebars(entries: list[LayerIdentifierEntry]) -> list[LayerIdentifierSidebar]:
    """Find the layer identifier column given the index column entries.

    Note: Similar to find_depth_columns.find_depth_columns. Refactoring may be desired.

    Args:
        entries (list[LayerIdentifierEntry]): The layer identifier column entries.

    Returns:
        list[LayerIdentifierSidebar]: The found layer identifier sidebar.
    """
    layer_identifier_sidebars = [LayerIdentifierSidebar([entries[0]])]
    for entry in entries[1:]:
        has_match = False
        for column in layer_identifier_sidebars:
            if column.can_be_appended(entry.rect):
                column.entries.append(entry)
                has_match = True
        if not has_match:
            layer_identifier_sidebars.append(LayerIdentifierSidebar([entry]))

        # only keep columns whose entries are not fully contained in a different column
        layer_identifier_sidebars = [
            column
            for column in layer_identifier_sidebars
            if all(not other.strictly_contains(column) for other in layer_identifier_sidebars)
        ]
        # check if the column rect is a subset of another column rect. If so, merge the entries and sort them by y0.
        for column in layer_identifier_sidebars:
            for other in layer_identifier_sidebars:
                if column != other and column.is_contained(other.rect()):
                    for entry in other.entries:
                        if entry not in column.entries:
                            column.entries.append(entry)
                    column.entries.sort(key=lambda entry: entry.rect.y0)
                    layer_identifier_sidebars.remove(other)
                    break
    layer_identifier_sidebars = [column for column in layer_identifier_sidebars if len(column.entries) > 2]
    return layer_identifier_sidebars


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
            layer_depth_entry = AToBDepthColumnEntry.from_text(line.text, line.rect, require_start_of_string=False)
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
