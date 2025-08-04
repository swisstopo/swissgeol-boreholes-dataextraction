"""Module for the layer identifier sidebars."""

import logging
import re
from dataclasses import dataclass

import numpy as np
import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textblock import TextBlock
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

from ...base.sidebar_entry import LayerIdentifierEntry
from ...interval.a_to_b_interval_extractor import AToBIntervalExtractor
from ...interval.interval import IntervalBlockGroup, IntervalBlockPair
from ...interval.partitions_and_sublayers import (
    get_optimal_intervals_with_text,
)
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
        provisional_groups = []
        last_end_depth = None
        for block in blocks:
            block_lines_header = self._get_header(block)  # Get the header lines from the block

            interval_block_pairs = AToBIntervalExtractor.from_material_description_lines(block.lines)
            interval_block_pairs = get_optimal_intervals_with_text(interval_block_pairs)

            ignored_lines = self._filter_header_lines(block_lines_header, interval_block_pairs)

            new_groups = [
                IntervalBlockGroup(
                    depth_intervals=[pair.depth_interval],
                    blocks=[self._clean_block(pair.block, ignored_lines)],
                )
                for pair in interval_block_pairs
            ]
            if (
                interval_block_pairs
                and interval_block_pairs[0].depth_interval
                and interval_block_pairs[-1].depth_interval
            ):
                new_start_depth = interval_block_pairs[0].depth_interval.start.value
                if new_start_depth != last_end_depth:
                    # only use this group without depth indications if the depths are discontinued
                    result.extend(provisional_groups)
                result.extend(new_groups)
                provisional_groups = []
                last_end_depth = (
                    interval_block_pairs[-1].depth_interval.end.value
                    if interval_block_pairs[-1].depth_interval.end
                    else None
                )
            else:
                # If we don't have a depth interval, then we only use this data if the start of the next depth interval
                # does not match the end of the last interval.
                # Like this, we avoid including headers such as "6) Retrait würmien" as a separate layer, even when
                # they have their own indicator in the profile.
                provisional_groups.extend(new_groups)
        result.extend(provisional_groups)

        return result

    def _get_header(self, block: TextBlock) -> list[TextLine]:
        """Get the header lines from a block.

        Header lines are defined as lines that overlap significantly with the layer identifiers.

        Args:
            block (TextBlock): The block from which to extract the header lines.

        Returns:
            list[TextLine]: A list of header lines that overlap significantly with the layer identifiers.
        """
        return [
            line
            for line in block.lines
            if any(y_overlap_significant_smallest(line.rect, identifier.rect, 0.8) for identifier in self.entries)
        ]

    def _filter_header_lines(
        self, block_lines_header: list[TextLine], interval_block_pairs: list[IntervalBlockPair]
    ) -> list[TextLine]:
        """Filters out header lines from the blocks based on the layer identifiers.

        Args:
            block_lines_header (list[TextLine]): The header lines.
            interval_block_pairs (list[IntervalBlockPair]): The list of interval block pairs to filter.

        Returns:
            list[TextLine]: A list of lines that should be ignored, such as headers or layer identifiers.
        """

        def _is_header_capitalized(header_lines: list[TextLine]) -> bool:
            """Check if the header lines are capitalized.

            We require 70% of the letters in the header to be uppercase, excluding the layer identifier itself.
            """
            text = "".join([line.text for line in header_lines])
            for identifier in self.entries:
                letters = [c for c in text.replace(identifier.value, "") if c.isalpha()]
                if letters and np.mean([c.isupper() for c in letters]) > 0.7:
                    return True
            return False

        other_lines = [
            line for pair in interval_block_pairs for line in pair.block.lines if line not in block_lines_header
        ]

        header_capitalized = _is_header_capitalized(block_lines_header)
        has_depth_info = any(pair.depth_interval for pair in interval_block_pairs)
        if not header_capitalized and (not has_depth_info or not other_lines):
            # If the header is not capitalized, and there is no other lines than the header, or no depth info are found
            # we treat the header as part of the description
            return []

        return block_lines_header

    def _clean_block(self, block: TextBlock, ignored_lines: list[TextLine]) -> TextBlock:
        """Remove the headers in ignored_lines and the layer identifiers from the block.

        Args:
            block (TextBlock): The block to clean.
            ignored_lines (list[TextLine]): The lines to ignore, such as headers.

        Returns:
            TextBlock: The cleaned block with the ignored lines and layer identifiers removed.
        """
        # Create set of entry values
        entry_values = {entry.value.strip() for entry in self.entries}

        new_word_lists = [
            [word for word in line.words if word.text.strip() not in entry_values]
            for line in block.lines
            if line not in ignored_lines
        ]

        # Only keep lines that have words remaining after filtering
        return TextBlock([TextLine(words) for words in new_word_lists if words])

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
