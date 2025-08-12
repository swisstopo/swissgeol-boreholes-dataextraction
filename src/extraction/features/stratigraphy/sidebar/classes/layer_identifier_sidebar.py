"""Module for the layer identifier sidebars."""

import logging
from dataclasses import dataclass

import numpy as np
import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textblock import TextBlock
from extraction.features.utils.text.textline import TextLine

from ...base.sidebar_entry import LayerIdentifierEntry
from ...interval.a_to_b_interval_extractor import AToBIntervalExtractor
from ...interval.interval import IntervalBlockGroup, IntervalBlockPair
from ...interval.partitions_and_sublayers import (
    get_optimal_intervals_with_text,
)
from .sidebar import Sidebar

logger = logging.getLogger(__name__)


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
        # skip the first layer identifier, and use None for the last block
        for next_layer_identifier in self.entries[1:] + [None]:
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

            ignored_lines = block_lines_header if self._ignore_header(block_lines_header, interval_block_pairs) else []

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

        Note: In rare cases where the header spans multiple lines, only the first line is considered the header.

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

    def _ignore_header(
        self, block_lines_header: list[TextLine], interval_block_pairs: list[IntervalBlockPair]
    ) -> bool:
        """Analyse whether to ignore header lines from the blocks based on the layer identifiers.

        Args:
            block_lines_header (list[TextLine]): The header lines.
            interval_block_pairs (list[IntervalBlockPair]): The list of interval block pairs to filter.

        Returns:
            bool: True if the header lines should be ignored, False otherwise.
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

        depths_in_header = bool(
            AToBIntervalExtractor.from_text(
                TextLine([word for line in block_lines_header for word in line.words]), require_start_of_string=False
            )
        )

        # If the header is capitalized, or there is other lines than the header and depth info are found in the header
        # we exclude the header from the material description.
        return bool(header_capitalized or (depths_in_header and other_lines))

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
