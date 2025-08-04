"""Module for the layer identifier sidebars."""

import logging
from dataclasses import dataclass

import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.geometry.util import y_overlap_significant_smallest
from extraction.features.utils.text.textblock import TextBlock
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

from ...base.sidebar_entry import LayerIdentifierEntry
from ...interval.a_to_b_interval_extractor import AToBIntervalExtractor
from ...interval.interval import IntervalBlockGroup
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

        def _is_header(line: TextLine) -> bool:
            return any(
                line.text.replace(identifier.value, "").isupper()
                and y_overlap_significant_smallest(line.rect, identifier.rect, 0.8)
                for identifier in self.entries
            )

        non_header_lines = [line for line in description_lines if not _is_header(line)]

        blocks = []
        line_index = 0
        for layer_identifier_idx, _layer_index in enumerate(self.entries):
            next_layer_identifier = (
                self.entries[layer_identifier_idx + 1] if layer_identifier_idx + 1 < len(self.entries) else None
            )

            matched_block = self.matching_blocks(non_header_lines, line_index, next_layer_identifier)
            line_index += sum([len(block.lines) for block in matched_block])
            blocks.extend(matched_block)

        result = []
        provisional_groups = []
        last_end_depth = None
        for block in blocks:
            interval_block_pairs = AToBIntervalExtractor.from_material_description_lines(block.lines)
            interval_block_pairs = get_optimal_intervals_with_text(interval_block_pairs)

            new_groups = [
                IntervalBlockGroup(depth_intervals=[pair.depth_interval], blocks=[pair.block])
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
                last_end_depth = interval_block_pairs[-1].depth_interval.end.value
            else:
                # If we don't have a depth interval, then we only use this data if the start of the next depth interval
                # does not match the end of the last interval.
                # Like this, we avoid including headers such as "6) Retrait würmien" as a separate layer, even when
                # they have their own indicator in the profile.
                provisional_groups.extend(new_groups)
        result.extend(provisional_groups)

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
