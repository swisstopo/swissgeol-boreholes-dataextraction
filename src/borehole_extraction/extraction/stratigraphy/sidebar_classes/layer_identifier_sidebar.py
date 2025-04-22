"""Module for the layer identifier sidebars."""

import logging
import re
from dataclasses import dataclass

import pymupdf
from borehole_extraction.extraction.util_extraction.geometry.geometry_dataclasses import Line
from borehole_extraction.extraction.util_extraction.text.textblock import TextBlock
from borehole_extraction.extraction.util_extraction.text.textline import TextLine
from general_utils.file_utils import read_params

from ..base_sidebar_entry.sidebar_entry import LayerIdentifierEntry
from ..interval.a_to_b_interval_extractor import AToBIntervalExtractor
from ..interval.interval import IntervalBlockGroup
from .sidebar import Sidebar

logger = logging.getLogger(__name__)
matching_params = read_params("matching_params.yml")


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
        material_description_rect: pymupdf.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Divide the description lines into blocks based on the layer identifier entries.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (pymupdf.Rect): The bounding box of the material description.
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
            depth_interval = AToBIntervalExtractor.from_material_description_lines(block.lines)
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

    def is_contained(self, rect: pymupdf.Rect) -> bool:
        """Check if the layer identifier column is contained in another rectangle.

        Args:
            rect (pymupdf.Rect): The rectangle to check if it contains the layer identifier column.

        Returns:
            bool: True if the layer identifier column is contained in the rectangle, False otherwise.
        """
        return (
            rect.x0 <= self.rect().x0
            and self.rect().x1 <= rect.x1
            and rect.y0 <= self.rect().y0
            and self.rect().y1 <= rect.y1
        )

    def _standardize_key(self, value: str) -> list[str | int]:
        """Splits a string into parts: [int, str, int] for natural sorting.

        Example: '6d12)' → [6, 'd', 12]

        Args:
            value (str): the value to convert to standartized key.

        Returns:
            list[str | int]: the list containing the key in order.
        """
        value = value.strip().replace(")", "").lower()

        # Split into alternating numbers and letters
        parts = re.findall(r"\d+|[a-z]+", value)
        # conversion to int needed because 2 < 12 for example (with strings, '12' < '2', because '1' < '2')
        key = [int(p) if p.isdigit() else p for p in parts]
        return key

    def has_regular_progression(self):
        """Checks if a LayerIdentifierSidebar object is valid.

        This check is particularly useful to reject cases where columns of heights are detected, due to the letter
            "m)" which could indicate meters (e.g., 350.0 m in deepwell Arsch). It also rejects some invalid sidebars
            that appear in other documents.

        Returns:
            bool: Indicates whether the sidebar follows a logical order.
        """
        valid_count = 0
        for entry, next_entry in zip(self.entries, self.entries[1:], strict=False):
            current = self._standardize_key(entry.value)
            next_ = self._standardize_key(next_entry.value)

            try:
                if current < next_:
                    valid_count += 1
            except TypeError as e:
                # happens when comparing a string with an int, it is usually due to a extraction error (e.g. "e" < 2)
                logger.warning(
                    f"{e} encountered during comparison between {entry.value} and {next_entry.value}. This is "
                    "typically due to extraction issues, such as mixing strings and numbers (l -> 1, q -> 9)."
                )
                continue

        valid_ratio = valid_count / (len(self.entries) - 1)
        return valid_ratio > matching_params["layer_identifier_acceptance_ratio"]
