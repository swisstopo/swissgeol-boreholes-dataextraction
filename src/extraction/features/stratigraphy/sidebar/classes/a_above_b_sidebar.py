"""Module for the AAboveBSidebar, where the depths of layer interfaces are defined above/below each other."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import numpy as np

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.interval.interval import IntervalZone
from extraction.features.stratigraphy.sidebar.classes.depth_column_entry_sidebar import DepthColumEntrySidebar
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine


@dataclass
class AAboveBSidebar(DepthColumEntrySidebar):
    """Represents a sidebar where the depths of the layer boundaries are displayed in a column, above each other.

    Usually, the vertical position of a depth label on the page is proportional to the depth value.

    Example:
        0m

        0.2m


        0.5m
        ...
    """

    kind: ClassVar[str] = "a_above_b"

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
        return min(new_columns, key=lambda column: column.linear_fit_loss())

    def linear_fit_loss(self) -> float:
        if len(self.entries) == 0:
            return 0

        # We look at the lower y coordinate, because most often the baseline of the depth value text is aligned with
        # the line of the corresponding layer boundary.
        positions = np.array([entry.rect.y1 for entry in self.entries])
        values = np.array([entry.value for entry in self.entries])

        if len(set(positions)) >= 2:
            b, a = np.polynomial.polynomial.polyfit(positions, values, 1)  # linear regression
        else:
            b, a = np.median(positions), 0
        squared_errors = [(entry.value - (a * entry.rect.y1 + b)) ** 2 for entry in self.entries]
        mean_squared_error = sum(squared_errors) / len(self.entries)
        return mean_squared_error

    def fix_ocr_mistakes(self):
        """Correct common OCR mistakes (e.g. missing decimal points) if it makes the values more plausible."""
        if not self.entries:
            return self

        def new_sidebar(index: int, new_value: float) -> AAboveBSidebar:
            entry = self.entries[index]
            new_entry = DepthColumnEntry(rect=entry.rect, value=new_value, page_number=entry.page_number)
            return AAboveBSidebar([*self.entries[:index], new_entry, *self.entries[index + 1 :]])

        def score(sidebar: AAboveBSidebar) -> tuple[int, float]:
            return (sidebar.ascending_count(), -sidebar.linear_fit_loss())

        best_score = score(self)
        best_index = None
        best_new_value = None

        continue_search = True
        # Repeatedly improve the values until there is nothing left to improve
        while continue_search:
            continue_search = False
            median_value = np.median(np.array([entry.value for entry in self.entries]))

            for i, entry in enumerate(self.entries):
                if entry.value.is_integer() and entry.value > median_value:
                    candidate_values = [entry.value / 100, entry.value / 10]

                    for new_value in candidate_values:
                        new_score = score(new_sidebar(i, new_value))
                        if new_score > best_score:
                            best_score = new_score
                            best_index = i
                            best_new_value = new_value

            if best_index is not None:
                continue_search = True
                old_entry = self.entries[best_index]
                self.entries[best_index] = DepthColumnEntry(
                    rect=old_entry.rect, value=best_new_value, page_number=old_entry.page_number
                )
                best_index = None
                best_new_value = None
        return self

    def ascending_count(self) -> int:
        """Count how many pairs of values are in ascending order."""
        count = 0
        for index1, entry1 in enumerate(self.entries):
            for entry2 in self.entries[index1 + 1 :]:
                if entry1.value < entry2.value:
                    count += 1
        return count

    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries.

        We pass the rectangles of the intervals shifted by the vertical amount inferred with the diagonal lines.
        The related interval is the extracted one, without shifting.

        Returns:
            list[IntervalZone]: A list of interval zones.
        """
        return [
            IntervalZone(
                interval.start.shifted_rect if interval.start else None,
                interval.end.shifted_rect if interval.end else None,
                interval,
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

        return (close_to_mid_zone_bonus + falls_inside_bonus) / 2  # mean between the two is a good tradeoff.

    def compute_entries_shift(self, diagonals: list[Line]):
        """Compute the vertical shift for each sidebar entry based on the diagonal lines.

        The shift indicates how much higher or lower the entry should be matched compared to its current position.
        To avoid mismatches, we only consider the closest diagonal line to each entry, and stop the process
        when the distance between the entry and the diagonal is too large compared to the average entry height.

        Note: this function modifies the attribute 'relative_shift' of each sidebar entry in place.

        Args:
            diagonals (list[Line]): The diagonal lines.
        """
        avg_entries_height = sum([entry.rect.height for entry in self.entries]) / len(self.entries)
        seen_diags = []
        seen_entries = []
        while len(seen_diags) != len(diagonals) and len(seen_entries) != len(self.entries):
            unseen_diags = [diag for diag in diagonals if diag not in seen_diags]
            unseen_entries = [entry for entry in self.entries if entry not in seen_entries]
            closest_entry, closest_diag = min(
                [(entry, diag) for entry in unseen_entries for diag in unseen_diags],
                key=lambda pair: abs((pair[0].rect.y0 + pair[0].rect.y1) / 2 - pair[1].start.y),
            )
            if abs((closest_entry.rect.y0 + closest_entry.rect.y1) / 2 - closest_diag.start.y) > avg_entries_height:
                break  # remaining pairs are likely wrong, due to undetected entry or duplicated diagonal detection
            closest_entry.relative_shift = float(closest_diag.end.y - closest_diag.start.y)
            seen_diags.append(closest_diag)
            seen_entries.append(closest_entry)  # different post processing is needed depending on sidebar type

    def prevent_shifts_crossing(self):
        """Ensure that the vertical shifts of sidebar entries do not cause them to cross each other.

        Note: this function modifies the attribute 'relative_shift' of each sidebar entry in place.
        """
        prev_y1 = 0.0
        for entry in self.entries:
            entry_y0 = entry.rect.y0
            entry_height = entry.rect.y1 - entry.rect.y0
            if prev_y1 - entry_height / 2 > entry_y0 + entry.relative_shift:
                entry.relative_shift = prev_y1 - entry_y0 - entry_height / 2
            prev_y1 = entry_y0 + entry.relative_shift + entry_height


def generate_alternatives(value: float) -> list[float]:
    """Generate a list of all possible alternatives by replacing each '4' with '1'."""
    value_str = str(value)
    alternatives = []
    options = [(char if char != "4" else ["4", "1"]) for char in value_str]

    for combo in product(*options):
        alternatives.append(float("".join(combo)))

    return alternatives
