"""Module for the spulprobe sidebars."""

import pymupdf
from extraction.features.stratigraphy.base.sidebar_entry import SpulprobeEntry
from extraction.features.stratigraphy.interval.interval import AAboveBInterval, IntervalBlockGroup
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.find_description import get_description_blocks
from extraction.features.utils.text.textline import TextLine


class SpulprobeSidebar(Sidebar[SpulprobeEntry]):
    """Spulprobe sidebar."""

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

        Example return value:
            [
                IntervalBlockGroup(
                    depth_intervals=[AAboveBInterval(None, 0.1), AAboveBInterval(0.1, 0.3), ...],
                    blocks=[TextBlock(...), TextBlock(...), ...]
                ),
                IntervalBlockGroup(
                    depth_intervals=[AAboveBInterval(0.3, 0.7)],
                    blocks=[TextBlock(...), TextBlock(...), ...]
                ),
                ...
            ]
        """
        depth_intervals = self.get_intervals()

        groups = []

        current_intervals = []
        current_blocks = []
        all_blocks = get_description_blocks(
            description_lines,
            geometric_lines,
            material_description_rect,
            params["block_line_ratio"],
            left_line_length_threshold=params["left_line_length_threshold"],
            target_layer_count=len(depth_intervals),
        )

        block_index = 0

        for interval in depth_intervals:
            # HERE TWEAK
            pre, exact, post = interval.matching_blocks(all_blocks, block_index, params["min_block_clearance"])
            block_index += len(pre) + len(exact) + len(post)

            current_blocks.extend(pre)
            if exact:
                if len(current_intervals) > 0 or len(current_blocks) > 0:
                    groups.append(IntervalBlockGroup(depth_intervals=current_intervals, blocks=current_blocks))
                groups.append(IntervalBlockGroup(depth_intervals=[interval], blocks=exact))
                current_blocks = post
                current_intervals = []
            else:
                # last open ended interval is conventional with Spulprobe
                current_intervals.append(interval)

        if len(current_intervals) > 0 or len(current_blocks) > 0:
            groups.append(IntervalBlockGroup(depth_intervals=current_intervals, blocks=current_blocks))

        return groups

    def get_intervals(self) -> list[AAboveBInterval]:
        """Creates a list of depth intervals from Spulprobe entries.

        The first depth interval begins with the first Sp. tag.

        Returns:
            list[AAboveBInterval]: A list of depth intervals.
        """
        depth_intervals = []
        # depth_intervals.append(AAboveBInterval(None, self.entries[0])) # NO interval from 0 to first Sp.
        for i in range(len(self.entries) - 1):
            depth_intervals.append(AAboveBInterval(self.entries[i], self.entries[i + 1]))
        depth_intervals.append(
            AAboveBInterval(self.entries[len(self.entries) - 1], None)
        )  # even though no open ended intervals are allowed, they are still useful for matching,
        # especially for documents where the material description rectangle is too tall
        # (and includes additional lines below the actual material descriptions).
        return depth_intervals
