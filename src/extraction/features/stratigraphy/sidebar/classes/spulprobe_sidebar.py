"""Module for the spulprobe sidebars."""

import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.interval.interval import IntervalBlockGroup, SpulprobeInterval
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.find_description import get_description_blocks
from extraction.features.utils.text.textline import TextLine


class SpulprobeSidebar(Sidebar[SpulprobeEntry]):
    """Spulprobe sidebar where entries are depths in the form `Sp. X m`."""

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: pymupdf.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to the Sp. tags.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (pymupdf.Rect): The bounding box of the material description.
            params (dict): A dictionary of relevant parameters.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.
        """
        depth_intervals = self.get_intervals()

        groups = []

        all_blocks = get_description_blocks(
            description_lines,
            geometric_lines,
            material_description_rect,
            params["block_line_ratio"],
            left_line_length_threshold=params["left_line_length_threshold"],
            target_layer_count=len(depth_intervals) - 1,  # first interval None->1st Sp. should not be match usually.
        )
        all_blocks.sort(key=lambda b: (b.rect.y0, b.rect.x0))

        block_index = 0
        current_blocks = []
        # we start with interval being the first Sp. interval (prev_interval is the open-ended starting interval)
        for prev_interval, interval in zip(depth_intervals, depth_intervals[1:], strict=False):
            pre, exact = interval.matching_blocks(all_blocks, block_index)
            block_index += len(pre) + len(exact)

            current_blocks.extend(pre)
            groups.append(IntervalBlockGroup(depth_intervals=[prev_interval], blocks=current_blocks))
            current_blocks = exact.copy()

        # pick up all the remaining blocks, and assign them to the last interval
        current_blocks.extend(all_blocks[block_index:])
        groups.append(IntervalBlockGroup(depth_intervals=[interval], blocks=current_blocks))
        return groups

    def get_intervals(self) -> list[SpulprobeInterval]:
        """Creates a list of depth intervals from Spulprobe entries.

        Returns:
            list[SpulprobeInterval]: A list of depth intervals.
        """
        intervals = []
        # we need the open-ended interval for blocks that began on the previous page and that should not be matched
        # to any Sp. entry on the current page.
        intervals.append(SpulprobeInterval(None, self.entries[0]))
        for i in range(len(self.entries) - 1):
            intervals.append(SpulprobeInterval(self.entries[i], self.entries[i + 1]))
        intervals.append(
            SpulprobeInterval(self.entries[len(self.entries) - 1], None)
        )  # even though no open ended intervals are allowed, they are still useful for matching,
        # especially for documents where the material description rectangle is too tall
        # (and includes additional lines below the actual material descriptions).
        return intervals
