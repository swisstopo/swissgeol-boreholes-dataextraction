"""This module contains functions to compute the affinity between adjacent lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymupdf

from extraction.features.utils.geometry.geometry_dataclasses import Line
from utils.file_utils import read_params

from .textline import TextLine

merging_params = read_params("line_detection_params.yml")["line_merging_params"]


@dataclass
class Affinity:
    """class holding the 4 types of affinities.

    Each affinity is a float between -1.0 and 1.0 and measures how likely two lines belong together.

    long_lines_affinity: affinity based on the presence of long horizontal lines inbetween the two text lines.
    lines_on_the_left_affinity: affinity based on the presence of lines cutting the left side of the text lines.
    vertical_spacing_affinity: affinity based on the vertical spacing between the two text lines.
    right_end_affinity: affinity based on the alignement of the right end of the two text lines.
    """

    long_lines_affinity: float
    lines_on_the_left_affinity: float
    vertical_spacing_affinity: float
    right_end_affinity: float

    @classmethod
    def get_zero_affinity(cls) -> Affinity:
        """Get an affinity with all zero values."""
        return cls(0.0, 0.0, 0.0, 0.0)

    def weighted_affinity(
        self, line_weight: float, left_line_weight: float, spacing_weight: float, right_end_weight: float
    ) -> float:
        """Compute the weighted affinity using the provided weights."""
        return (
            line_weight * self.long_lines_affinity
            + left_line_weight * self.lines_on_the_left_affinity
            + spacing_weight * self.vertical_spacing_affinity
            + right_end_weight * self.right_end_affinity
        )

    def total_affinity(self) -> float:
        """Compute the total affinity with equal weights."""
        return self.weighted_affinity(1.0, 1.0, 1.0, 1.0)


class LineAffinityCalculator:
    """Class responsible to compute the line affinities."""

    def __init__(
        self,
        block_line_ratio: float,
        left_line_length_threshold: float,
        material_description_rect: pymupdf.Rect,
        geometric_lines: list[Line],
        description_lines: list[TextLine],
    ):
        """Initialize the LineAffinityCalculator.

        Args:
            block_line_ratio (float): Percentage of the block width that needs to be covered by a line.
            left_line_length_threshold (int): The minimum length of a line segment on the left side of a block to
                split it.
            material_description_rect (pymupdf.Rect): The bounding box for all material descriptions.
            geometric_lines (list[Line]): The geometric lines detected in the pdf page.
            description_lines (list[TextLine]): The list of description lines
        """
        self.block_line_ratio = block_line_ratio
        self.left_line_length_threshold = left_line_length_threshold

        self.material_description_rect = material_description_rect
        horizontal_slope_tolerance = merging_params["horizontal_slope_tolerance"]
        horizontal_lines = [line for line in geometric_lines if line.is_horizontal(horizontal_slope_tolerance)]

        # spanning horizontals
        self.lines_spanning_description = [line for line in horizontal_lines if self._is_spanning_description(line)]

        # left hand side splitter lines
        self.long_horizontals = [line for line in horizontal_lines if self._is_long_enough(line)]

        # vertical spacing
        self.description_lines = description_lines
        distances = [
            line.rect.y0 - prev_line.rect.y0
            for prev_line, line in zip(description_lines, description_lines[1:], strict=False)
            if line.rect.y0 > prev_line.rect.y0 + prev_line.rect.height / 2
        ]
        self.spacing_threshold = min(distances) * 1.15 if distances else None

    def _is_spanning_description(self, line: Line) -> bool:
        """Check if a line spans the material description rectangle.

        Args:
            line (Line): The line to check.

        Returns:
            bool: True if the line spans the material description rectangle, False otherwise.
        """
        line_left_x = np.min([line.start.x, line.end.x])
        line_right_x = np.max([line.start.x, line.end.x])
        return (
            np.min([self.material_description_rect.x1, line_right_x])
            - np.max([self.material_description_rect.x0, line_left_x])
            > self.block_line_ratio * self.material_description_rect.width
        )

    def _is_long_enough(self, line: Line) -> bool:
        """Check if a line is long enough to split a block on the left side.

        Args:
            line (Line): The line to check.

        Returns:
            bool: True if the line is long enough, False otherwise.
        """
        return np.abs(line.start.x - line.end.x) > self.left_line_length_threshold

    def compute_long_lines_affinity(self, previous_line: TextLine, current_line: TextLine) -> float:
        """Check if a block is separated by a line.

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, 0.0 otherwise.
        """
        last_line_y_coordinate = (previous_line.rect.y0 + previous_line.rect.y1) / 2
        current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2
        for line in self.lines_spanning_description:
            line_y_coordinate = (line.start.y + line.end.y) / 2
            if last_line_y_coordinate < line_y_coordinate < current_line_y_coordinate:
                return -1.0
        return 0.0

    def compute_lines_on_the_left_affinity(self, previous_line: TextLine, current_line: TextLine) -> float:
        """Check if a block is separated by a line segment on the left side of the block.

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, 0.0 otherwise.
        """
        last_line_y_coordinate = (previous_line.rect.y0 + previous_line.rect.y1) / 2
        current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2
        for line in self.long_horizontals:
            line_y_coordinate = (line.start.y + line.end.y) / 2
            geom_line_is_in_between = last_line_y_coordinate < line_y_coordinate < current_line_y_coordinate

            SLACK = 2  # pixels
            line_cuts_lefthandside_of_text = (
                line.start.x - SLACK < previous_line.rect.x0 < line.end.x + SLACK
                or line.start.x - SLACK < current_line.rect.x0 < line.end.x + SLACK
            )

            if geom_line_is_in_between and line_cuts_lefthandside_of_text:
                return -1.0
        return 0.0

    def compute_vertical_spacing_affinity(self, previous_line: TextLine, current_line: TextLine) -> float:
        """Check if a block is separated by sufficient vertical space.

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, up to 1.0 if they totally overlap.
        """
        current_rect = current_line.rect
        previous_rect = previous_line.rect
        return max(-1.0, 1.0 - (current_rect.y1 - previous_rect.y1) / current_rect.height)

    def compute_right_end_affinity(self, previous_line: TextLine, current_line: TextLine) -> float:
        """Check the alignment of the right end of the lines.

        If the previous line ends before the current one, it is unlikelly that it is the continuation of the
        description. It would look something like that:

            This is the
            description of a layer.

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, 0.0 otherwise.
        """
        return max(-1.0, min(0.0, -(current_line.rect.x1 - previous_line.rect.x1) / current_line.rect.width))

    def _indentation_affinity(self, previous_line: TextLine, current_line: TextLine):
        """Split the text block based on indentation.

        note: not currently used. is_indented attribute was removed from TextLine, put back if needed.

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, 0.0 otherwise.
        """
        # indentation
        prev_line_start = previous_line.rect.x0
        current_line_start = current_line.rect.x0
        max_line_width = max([line.rect.width for line in (previous_line, current_line)])

        low_margin = 0.03 * max_line_width
        high_margin = 0.2 * max_line_width

        if previous_line.is_indented:  # attribute was removed
            if current_line_start >= prev_line_start - low_margin:
                current_line.is_indented = True
                return 0.0  # both lines are indented
            else:
                return -1.0  # previous line was indented, this one is not

        if prev_line_start + low_margin <= current_line_start <= prev_line_start + high_margin:
            current_line.is_indented = True
        return 0.0

    def _unindentation_affinity(self, previous_line: TextLine, current_line: TextLine):
        """Split the text block based on indentation return (end of indentation block).

        note: not currently used

        Args:
            previous_line (TextLine): The previous line.
            current_line (TextLine): The current line.

        Returns:
            float: The affinity: -1.0 if lines are not compatible, 0.0 otherwise.
        """
        # indentation
        prev_line_start = previous_line.rect.x0
        current_line_start = current_line.rect.x0
        max_line_width = max([line.rect.width for line in (previous_line, current_line)])

        low_margin = 0.03 * max_line_width

        return -1.0 if current_line_start < prev_line_start - low_margin else 0.0


def get_line_affinity(
    description_lines: list[TextLine],
    material_description_rect: pymupdf.Rect,
    geometric_lines: list[Line],
    block_line_ratio: float,
    left_line_length_threshold: float,
):
    """Compute the affinity of each line with the previous one, base of the presence of horizontal line inbetween.

    Args:
        block_line_ratio (float): Percentage of the block width that needs to be covered by a line.
        left_line_length_threshold (int): The minimum length of a line segment on the left side of a block to split it.
        material_description_rect (pymupdf.Rect): The bounding box for all material descriptions.
        geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        description_lines (list[TextLine]): The list of description lines
    """
    calculator = LineAffinityCalculator(
        block_line_ratio, left_line_length_threshold, material_description_rect, geometric_lines, description_lines
    )
    affinities = [Affinity.get_zero_affinity()]  # first line has no previous
    for previous_line, line in zip(description_lines, description_lines[1:], strict=False):
        affinities.append(
            Affinity(
                calculator.compute_long_lines_affinity(previous_line, line),
                calculator.compute_lines_on_the_left_affinity(previous_line, line),
                calculator.compute_vertical_spacing_affinity(previous_line, line),
                calculator.compute_right_end_affinity(previous_line, line),
            )
        )

    return affinities
