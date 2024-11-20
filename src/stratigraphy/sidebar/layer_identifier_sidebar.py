"""Module for the layer identifier sidebars."""

from dataclasses import dataclass

import fitz

from stratigraphy.lines.line import TextLine
from stratigraphy.text.textblock import TextBlock
from stratigraphy.util.dataclasses import Line

from ..util.a_to_b_interval_extractor import AToBIntervalExtractor
from .interval_block_group import IntervalBlockGroup
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
            depth_interval = AToBIntervalExtractor.from_lines(block.lines)
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
