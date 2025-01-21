"""Module for the AAboveBSidebar, where the depths of layer interfaces are defined above/below each other."""

from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np

from stratigraphy.depth import AAboveBInterval
from stratigraphy.lines.line import TextLine
from stratigraphy.text.find_description import get_description_blocks
from stratigraphy.util.dataclasses import Line

from .interval_block_group import IntervalBlockGroup
from .sidebar import Sidebar
from .sidebarentry import DepthColumnEntry


@dataclass
class AAboveBSidebar(Sidebar[DepthColumnEntry]):
    """Represents a sidebar where the depths of the layer boundaries are displayed in a column, above each other.

    Usually, the vertical position of a depth label on the page is proportional to the depth value.

    Example:
        0m

        0.2m


        0.5m
        ...
    """

    entries: list[DepthColumnEntry]

    def __repr__(self):
        return "AAboveBSidebar({})".format(", ".join([str(entry) for entry in self.entries]))

    def strictly_contains(self, other: AAboveBSidebar) -> bool:
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_strictly_increasing(self) -> bool:
        return all(i.value < j.value for i, j in zip(self.entries, self.entries[1:], strict=False))

    def depth_intervals(self) -> list[AAboveBInterval]:
        """Creates a list of depth intervals from the depth column entries.

        The first depth interval has an open start value (i.e. None).

        Returns:
            list[AAboveBInterval]: A list of depth intervals.
        """
        depth_intervals = [AAboveBInterval(None, self.entries[0])]
        for i in range(len(self.entries) - 1):
            depth_intervals.append(AAboveBInterval(self.entries[i], self.entries[i + 1]))
        depth_intervals.append(
            AAboveBInterval(self.entries[len(self.entries) - 1], None)
        )  # even though no open ended intervals are allowed, they are still useful for matching,
        # especially for documents where the material description rectangle is too tall
        # (and includes additional lines below the actual material descriptions).
        return depth_intervals

    def significant_arithmetic_progression(self) -> bool:
        # to allow for OCR errors or gaps in the progression, we only require a segment of length 6 that is an
        # arithmetic progression
        segment_length = 6
        if len(self.entries) < segment_length:
            return self.is_arithmetic_progression()
        else:
            for i in range(len(self.entries) - segment_length + 1):
                if AAboveBSidebar(self.entries[i : i + segment_length]).is_arithmetic_progression():
                    return True
            return False

    def is_arithmetic_progression(self) -> bool:
        if len(self.entries) <= 2:
            return True

        progression = np.array(range(len(self.entries)))
        entries = np.array([entry.value for entry in self.entries])

        # Avoid warnings in the np.corrcoef call, as the correlation coef is undefined if the standard deviation is 0.
        if np.std(entries) == 0:
            return False

        scale_pearson_correlation_coef = np.corrcoef(entries, progression)[0, 1].item()
        return abs(scale_pearson_correlation_coef) >= 0.9999

    def pearson_correlation_coef(self) -> float:
        # We look at the lower y coordinate, because most often the baseline of the depth value text is aligned with
        # the line of the corresponding layer boundary.
        positions = np.array([entry.rect.y1 for entry in self.entries])
        entries = np.array([entry.value for entry in self.entries])

        # Avoid warnings in the np.corrcoef call, as the correlation coef is undefined if the standard deviation is 0.
        if np.std(entries) == 0 or np.std(positions) == 0:
            return 0

        return np.corrcoef(positions, entries)[0, 1].item()

    def remove_entry_by_correlation_gradient(self) -> AAboveBSidebar | None:
        if len(self.entries) < 3:
            return None

        new_columns = [
            AAboveBSidebar([entry for index, entry in enumerate(self.entries) if index != remove_index])
            for remove_index in range(len(self.entries))
        ]
        return max(new_columns, key=lambda column: column.pearson_correlation_coef())

    def make_ascending(self):
        median_value = np.median(np.array([entry.value for entry in self.entries]))
        for i, entry in enumerate(self.entries):
            if entry.value.is_integer() and entry.value > median_value:
                factor100_value = entry.value / 100
                previous_ok = i == 0 or self.entries[i - 1].value < factor100_value
                next_ok = i + 1 == len(self.entries) or factor100_value < self.entries[i + 1].value

                if previous_ok and next_ok:
                    # Create a new entry instead of modifying the value of the current one, as this entry might be
                    # used in different sidebars as well.
                    self.entries[i] = DepthColumnEntry(rect=entry.rect, value=factor100_value)

        return self

    def break_on_double_descending(self) -> list[AAboveBSidebar]:
        segments = []
        segment_start = 0
        for index, current_entry in enumerate(self.entries):
            if (
                index >= 2
                and index + 1 < len(self.entries)
                and current_entry.value < self.entries[index - 2].value
                and current_entry.value < self.entries[index - 1].value
                and self.entries[index + 1].value < self.entries[index - 2].value
                and self.entries[index + 1].value < self.entries[index - 1].value
            ):
                # big big || small small
                segments.append(self.entries[segment_start:index])
                segment_start = index

        final_segment = self.entries[segment_start:]
        if final_segment:
            segments.append(final_segment)

        return [AAboveBSidebar(segment) for segment in segments]

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: fitz.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Note: includes a heuristic of whether there should be a group corresponding to a final depth interval
        starting from the last depth entry without any end value.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.
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
        depth_intervals = self.depth_intervals()

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
            # don't allow a layer above depth 0
            if interval.start is None and interval.end.value == 0:
                continue

            pre, exact, post = interval.matching_blocks(all_blocks, block_index)
            block_index += len(pre) + len(exact) + len(post)

            current_blocks.extend(pre)
            if len(exact):
                if len(current_intervals) > 0 or len(current_blocks) > 0:
                    groups.append(IntervalBlockGroup(depth_intervals=current_intervals, blocks=current_blocks))
                groups.append(IntervalBlockGroup(depth_intervals=[interval], blocks=exact))
                current_blocks = post
                current_intervals = []
            else:
                # The final open-ended interval should not be added, since borehole profiles do typically not come
                # with open-ended intervals.
                if interval.end is not None:
                    current_intervals.append(interval)

        if len(current_intervals) > 0 or len(current_blocks) > 0:
            groups.append(IntervalBlockGroup(depth_intervals=current_intervals, blocks=current_blocks))

        return groups
