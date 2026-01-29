"""Module for the layer identifier sidebars."""

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from extraction.features.stratigraphy.base.sidebar_entry import LayerIdentifierEntry
from extraction.features.stratigraphy.interval.a_to_b_interval_extractor import AToBIntervalExtractor
from extraction.features.stratigraphy.interval.interval import IntervalBlockPair, IntervalZone
from extraction.features.stratigraphy.interval.partitions_and_sublayers import (
    get_optimal_intervals_with_text,
)
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from swissgeol_doc_processing.geometry.util import y_overlap_significant_smallest
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine

logger = logging.getLogger(__name__)


@dataclass
class LayerIdentifierSidebar(Sidebar[LayerIdentifierEntry]):
    """Class for a layer identifier sidebar.

    Layer identifiers are labels that are particularly common in Deriaz layout borehole profiles. They can be
    sequential such as in 1007.pdf - a), b), c), etc. - or contain some semantic meaning such as in 10781.pdf -
    5c12), 4a), etc.
    """

    entries: list[LayerIdentifierEntry]

    kind: ClassVar[str] = "layer_identifier"

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        if not self.entries:
            return []
        return self.get_zones_from_entries(self.entries, include_open_ended=True)

    @staticmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines to interval zones.

        The score is 1.0 if the line is located within the interval zone, 0.0 otherwise.
        For layer identifier sidebar, the zone begins and ends at the top of each rectangle bounds.

        Args:
            interval_zone (IntervalZone): The interval zone to score against.
            line (TextLine): The text line to score.

        Returns:
            float: The score for the given interval zone and text line.
        """
        return Sidebar.default_score(interval_zone, line)

    def post_processing(
        self, interval_lines_mapping: list[tuple[IntervalZone, list[TextLine]]]
    ) -> list[IntervalBlockPair]:
        """Post-process the matched interval zones and description lines into IntervalBlockPairs.

        For LayerIdentifierSidebars, we extract depth information from the text blocks because the depth is not
        contained in the sidebar entries.

        Args:
            interval_lines_mapping (list[tuple[IntervalZone, list[TextLine]]]): The matched interval zones and
                description lines.

        Returns:
            list[IntervalBlockPair]: The processed interval block pairs.
        """
        blocks = [TextBlock(lines) for _, lines in interval_lines_mapping if lines]
        return self.create_pairs_from_layer_identifier_blocks(blocks)

    def create_pairs_from_layer_identifier_blocks(self, blocks: list[TextBlock]) -> list[IntervalBlockPair]:
        """From the identified TextBlocks, extract depth information in the text and create IntervalBlockPairs.

        For each TextBlock, depth intervals are extracted from the text lines. Further post-processing is applied to
        remove headers and clean the text blocks.

        Args:
            blocks (list[TextBlock]): The list of text blocks to process.

        Returns:
            list[IntervalBlockPair]: The processed interval block pairs.
        """
        result = []
        provisional_pairs = []
        last_end_depth = None
        for block in blocks:
            block_lines_header = self._get_header(block)  # Get the header lines from the block

            interval_block_pairs = self._extract_intervals_from_lines(block.lines)
            interval_block_pairs = get_optimal_intervals_with_text(interval_block_pairs)

            ignored_lines = block_lines_header if self._ignore_header(block_lines_header, interval_block_pairs) else []
            interval_block_pairs = [
                IntervalBlockPair(pair.depth_interval, self._clean_block(pair.block, ignored_lines))
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
                    result.extend(provisional_pairs)
                result.extend(interval_block_pairs)
                provisional_pairs = []
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
                provisional_pairs.extend(interval_block_pairs)
        result.extend(provisional_pairs)

        return result

    def _extract_intervals_from_lines(self, lines: list[TextLine]) -> list[IntervalBlockPair]:
        """Extract depth interval from text lines from a material description.

        For borehole profiles using the Deriaz layout, depth intervals are typically embedded within the material
        description text. These descriptions often further subdivide into multiple sublayers, each with its own
        distinct depth interval. This function extracts all such depth intervals found in the description, along with
        their corresponding text blocks. Decisions about which intervals to keep or discard are handled by downstream
        processing.
        For example (from GeoQuat 12306):
            1) REMBLAIS HETEROGENES
               0.00 - 0.08 m : Revêtement bitumineux
               0.08- 0.30 m : Grave d'infrastructure
               0.30 - 1.40 m : Grave dans importante matrice de sable
                               moyen, brun beige, pulvérulent.
        From this material description, this method will extract all depth intervals.

        Args:
            lines (list[TextLine]): The lines to extract the depth interval from.

        Returns:
            list[IntervalBlockPair]: a list of interval-block-pairs that can be extracted from the given lines
        """
        entries = []
        current_block = []
        current_interval = None
        start_depth = None
        prev_line = None
        prev_interval = None
        for idx, line in enumerate(lines):
            a_to_b_interval, line_without_depths = AToBIntervalExtractor.from_text(line, require_start_of_string=False)
            # First line of the block is stripped of its (potential) leading depths
            final_line = line if idx != 0 else line_without_depths
            if prev_line and not a_to_b_interval and not prev_interval:
                # if depth was not found in the previous and current lines, we look for a depth wrapping arround.
                combined_lines = TextLine(prev_line.words + line.words)
                a_to_b_interval, _ = AToBIntervalExtractor.from_text(combined_lines, require_start_of_string=False)
            prev_interval = a_to_b_interval
            prev_line = line
            # require_start_of_string = False because the depth interval may not always start at the beginning
            # of the line e.g. "Remblais Heterogene: 0.00 - 0.5m"
            if a_to_b_interval:
                # We assume that the first depth encountered is the start depth, and we reject further depth values
                # smaller than this first one. This avoids some false positives (e.g. GeoQuat 3339.pdf).
                if not start_depth:
                    start_depth = a_to_b_interval.start.value
                if a_to_b_interval.start.value >= start_depth:
                    if current_interval:
                        entries.append(IntervalBlockPair(current_interval, TextBlock(current_block)))
                        current_block = []
                        final_line = line_without_depths  # start of a new depth block, strip leading depth
                    current_interval = a_to_b_interval
            if final_line and final_line.words:
                current_block.append(final_line)
        if current_block:
            entries.append(IntervalBlockPair(current_interval, TextBlock(current_block)))

        return entries

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

        other_lines_presence = any(
            [line not in block_lines_header for pair in interval_block_pairs for line in pair.block.lines]
        )

        header_capitalized = _is_header_capitalized(block_lines_header)

        depths_in_header = bool(
            AToBIntervalExtractor.from_text(
                TextLine([word for line in block_lines_header for word in line.words]), require_start_of_string=False
            )[0]
        )

        # If the header is capitalized, or there is other lines than the header and depth info are found in the header
        # we exclude the header from the material description.
        return header_capitalized or (depths_in_header and other_lines_presence)

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
