"""Module for the AAboveBSidebar, where the depths of layer interfaces are defined above/below each other."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar

import numpy as np

from extraction.features.stratigraphy.interval.interval import AAboveBInterval, IntervalBlockPair, IntervalZone
from extraction.features.utils.text.textblock import TextBlock
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

    kind: ClassVar[str] = "a_above_b"

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
        depth_intervals = [] if self.entries[0].value == 0.0 else [AAboveBInterval(None, self.entries[0])]
        for i in range(len(self.entries) - 1):
            depth_intervals.append(AAboveBInterval(self.entries[i], self.entries[i + 1]))
        depth_intervals.append(AAboveBInterval(self.entries[-1], None))  # include open-ended
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
        if not self.entries:
            return self

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
                    self.entries[i] = DepthColumnEntry(rect=entry.rect, value=new_value, page_number=entry.page_number)
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

    def get_interval_zone(self):
        """Get the interval zones defined by the sidebar entries.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        return [
            IntervalZone(
                interval.start.rect if interval.start else None, interval.end.rect if interval.end else None, interval
            )
            for interval in self.depth_intervals()
        ]

    @staticmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines to AAboveBInterval zones.

        The score ranges from 1.0 to 0.0 and is composed of two parts.
        A first part that ranges from 0.5 to 0.0, based on how close the line is to the middle of the interval zone.
        And a bonus of 0.5 if the line falls completely inside the interval zone. For AaboveB sidebar, the zone
        begins and ends at the middle of each rectangle bounds, and the line's rectangle must be fully contained
        within the zone.

        Args:
            interval_zone (IntervalZone): The interval zone to score against.
            line (TextLine): The text line to score.

        Returns:
            float: The score from 0 to 1, for the given interval zone and text line.
        """
        start_mid = ((interval_zone.start.y0 + interval_zone.start.y1) / 2) if interval_zone.start else None
        end_mid = ((interval_zone.end.y0 + interval_zone.end.y1) / 2) if interval_zone.end else None
        line_mid = (line.rect.y0 + line.rect.y1) / 2
        falls_inside_bonus = 0.0
        if (start_mid is None or line.rect.y0 > start_mid) and (end_mid is None or line.rect.y1 < end_mid):
            falls_inside_bonus = 1.0  # textline is inside the depth interval

        if not (interval_zone.end and interval_zone.start):
            entry_mid = start_mid if start_mid else end_mid
            close_to_mid_zone_bonus = math.exp(-(abs(entry_mid - line_mid) / 30.0)) if entry_mid else 0.0
        else:
            mid_zone = (interval_zone.end.y0 + interval_zone.start.y1) / 2
            close_to_mid_zone_bonus = math.exp(-(abs(mid_zone - line_mid) / 30.0))  # 1 -> 0

        score = (close_to_mid_zone_bonus + falls_inside_bonus) / 2  # mean between the two is a good tradeoff.

        OPEN_ENDED_PENALTY = float("inf")  # penalize assigning lines to open ended (open-ended has no end)
        return score if end_mid else score - OPEN_ENDED_PENALTY

    def post_processing(
        self, interval_lines_mapping: list[tuple[IntervalZone, list[TextLine]]]
    ) -> list[IntervalBlockPair]:
        """Post-process the matched interval zones and description lines into IntervalBlockPairs."""
        # remove open-ended interval, but keep the lines
        return [
            IntervalBlockPair(zone.related_interval if zone.end else None, TextBlock(lines))
            for zone, lines in interval_lines_mapping
        ]


def generate_alternatives(value: float) -> list[float]:
    """Generate a list of all possible alternatives by replacing each '4' with '1'."""
    value_str = str(value)
    alternatives = []
    options = [(char if char != "4" else ["4", "1"]) for char in value_str]

    for combo in product(*options):
        alternatives.append(float("".join(combo)))

    return alternatives
