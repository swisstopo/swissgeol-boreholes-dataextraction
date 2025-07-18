"""Module for the AAboveBSidebar, where the depths of layer interfaces are defined above/below each other."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pymupdf

from extraction.features.stratigraphy.interval.interval import AAboveBInterval, IntervalBlockGroup
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.find_description import get_description_blocks
from extraction.features.utils.text.textline import TextLine

from ...base.sidebar_entry import DepthColumnEntry
from .sidebar import Sidebar


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
    skipped_entries: list[DepthColumnEntry] = field(default_factory=list)

    def __repr__(self):
        return "AAboveBSidebar({})".format(", ".join([str(entry) for entry in self.entries]))

    def strictly_contains(self, other: AAboveBSidebar) -> bool:
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_strictly_increasing(self) -> bool:
        return all(self.entries[i].value < self.entries[i + 1].value for i in range(len(self.entries) - 1))

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

    @staticmethod
    def is_close_to_arithmetic_progression(entries: list[DepthColumnEntry]) -> bool:
        """Check if entries are very close to an arithmetic progression."""
        if len(entries) <= 2:
            return False

        values = [entry.value for entry in entries]

        differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        step = round(statistics.median(differences), 2)
        if step <= 0:
            return False

        first = values[0]
        last = values[-1]
        arithmetic_progression = {
            # ensure we have nicely rounded numbers, without inaccuracies from floating point arithmetic
            round(value * step, 2)
            for value in range(int(first / step), int(last / step) + 1)
        }
        score = [value in arithmetic_progression for value in values].count(True)
        # 80% of the values must be contained in the closest arithmetic progression (allowing for 20% OCR errors)
        return score > 0.8 * len(values)

    def close_to_arithmetic_progression(self) -> bool:
        """Check if the depth values of the entries of this sidebar are very close to an arithmetic progression."""
        return AAboveBSidebar.is_close_to_arithmetic_progression(self.entries)

    def pearson_correlation_coef(self) -> float:
        # We look at the lower y coordinate, because most often the baseline of the depth value text is aligned with
        # the line of the corresponding layer boundary.
        positions = np.array([entry.rect.y1 for entry in self.entries])
        entries = np.array([entry.value for entry in self.entries])

        std_positions = np.std(positions)
        std_entries = np.std(entries)
        if std_positions == 0 or std_entries == 0:
            return 0

        # We calculate the Pearson correlation coefficient manually
        # to avoid redundant standard deviation calculations that would occur with np.corrcoef.
        covariance = np.mean((positions - np.mean(positions)) * (entries - np.mean(entries)))
        return covariance / (std_positions * std_entries)

    def remove_entry_by_correlation_gradient(self) -> AAboveBSidebar | None:
        if len(self.entries) < 3:
            return None

        new_columns = [
            AAboveBSidebar([entry for index, entry in enumerate(self.entries) if index != remove_index])
            for remove_index in range(len(self.entries))
        ]
        return max(new_columns, key=lambda column: column.pearson_correlation_coef())

    def remove_integer_scale(self):
        """Removes arithmetically progressing integers from this sidebar, as they are likely a scale."""
        integer_entries = [entry for entry in self.entries if not entry.has_decimal_point]
        if integer_entries and AAboveBSidebar.is_close_to_arithmetic_progression(integer_entries):
            self.skipped_entries = integer_entries
            self.entries = [entry for entry in self.entries if entry not in integer_entries]
        return self

    def make_ascending(self):
        """Adjust entries in this sidebar for an ascending order."""
        median_value = np.median(np.array([entry.value for entry in self.entries]))

        for i, entry in enumerate(self.entries):
            new_values = []

            if entry.value.is_integer() and entry.value > median_value:
                new_values.extend([entry.value / 100, entry.value / 10])

            # Correct common OCR mistakes where "4" is recognized instead of "1"
            # We don't control for OCR mistakes recognizing "9" as "3" (example zurich/680244005-bp.pdf)
            if "4" in str(entry.value) and not self._valid_value(i, entry.value):
                new_values.extend(generate_alternatives(entry.value))

            # Assign the first valid correction
            for new_value in new_values:
                if self._valid_value(i, new_value):
                    self.entries[i] = DepthColumnEntry(rect=entry.rect, value=new_value)
                    break
        return self

    def _valid_value(self, index: int, new_value: float) -> bool:
        """Check if new value at given index is maintaining ascending order."""
        previous_ok = index == 0 or all(other_entry.value < new_value for other_entry in self.entries[:index])
        next_ok = index + 1 == len(self.entries) or new_value < self.entries[index + 1].value
        return previous_ok and next_ok

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

        return [AAboveBSidebar(segment, self.skipped_entries) for segment in segments]

    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: pymupdf.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Note: includes a heuristic of whether there should be a group corresponding to a final depth interval
        starting from the last depth entry without any end value.

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

            pre, exact, post = interval.matching_blocks(all_blocks, block_index, params["min_block_clearance"])
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


def generate_alternatives(value: float) -> list[float]:
    """Generate a list of all possible alternatives by replacing each '4' with '1'."""
    value_str = str(value)
    alternatives = []
    options = [(char if char != "4" else ["4", "1"]) for char in value_str]

    for combo in product(*options):
        alternatives.append(float("".join(combo)))

    return alternatives
