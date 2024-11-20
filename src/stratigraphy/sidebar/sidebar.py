"""This module contains the Sidebar class, used to represent a depth column (or similar) of a borehole profile."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

import fitz

from stratigraphy.depthcolumnentry import DepthColumnEntry
from stratigraphy.lines.line import TextLine, TextWord
from stratigraphy.sidebar.interval_block_group import IntervalBlockGroup
from stratigraphy.util.dataclasses import Line

EntryT = TypeVar("EntryT", bound=DepthColumnEntry)


@dataclass
class Sidebar(abc.ABC, Generic[EntryT]):
    """Abstract Sidebar class, representing depths or other data displayed to the side of material descriptions."""

    entries: list[EntryT]

    def rects(self) -> list[fitz.Rect]:
        """Get the rectangles of the depth column entries."""
        return [entry.rect for entry in self.entries]

    def rect(self) -> fitz.Rect:
        """Get the bounding box of the depth column entries."""
        x0 = min([rect.x0 for rect in self.rects()])
        x1 = max([rect.x1 for rect in self.rects()])
        y0 = min([rect.y0 for rect in self.rects()])
        y1 = max([rect.y1 for rect in self.rects()])
        return fitz.Rect(x0, y0, x1, y1)

    @property
    def max_x0(self) -> float:
        """Get the maximum x0 value of the depth column entries."""
        return max([rect.x0 for rect in self.rects()])

    @property
    def min_x1(self) -> float:
        """Get the minimum x1 value of the depth column entries."""
        return min([rect.x1 for rect in self.rects()])

    def noise_count(self, all_words: list[TextWord]) -> int:
        """Counts the number of words that intersect with the depth column entries.

        Returns the number of words that intersect with the depth column entries, but are not part of the depth column.

        Args:
            all_words (list[TextWord]): A list of all text lines on the page.

        Returns:
            int: The number of words that intersect with the depth column entries but are not part of it.
        """

        def significant_intersection(other_rect):
            intersection = fitz.Rect(other_rect).intersect(self.rect())
            return intersection.is_valid and intersection.width > 0.25 * self.rect().width

        return len([word for word in all_words if significant_intersection(word.rect)]) - len(self.entries)

    @abc.abstractmethod
    def identify_groups(
        self,
        description_lines: list[TextLine],
        geometric_lines: list[Line],
        material_description_rect: fitz.Rect,
        **params,
    ) -> list[IntervalBlockGroup]:
        """Identifies groups of description blocks that correspond to depth intervals.

        Args:
            description_lines (list[TextLine]): A list of text lines that are part of the description.
            geometric_lines (list[Line]): A list of geometric lines that are part of the description.
            material_description_rect (fitz.Rect): The bounding box of the material description.
            params (dict): A dictionary of relevant parameters.

        Returns:
            list[IntervalBlockGroup]: A list of groups, where each group is a IntervalBlockGroup.
        """
        pass

    def can_be_appended(self, rect: fitz.Rect) -> bool:
        """Checks if a new depth column entry can be appended to the current depth column.

        Check if the middle of the new rect is between the outer horizontal boundaries of the column, and if there is
        an intersection with the minimal horizontal boundaries of the column.

        The checks are:
        - The width of the new rectangle is greater than the width of the current depth column. Or;
        - The middle of the new rectangle is within the horizontal boundaries of the current depth column.
        - The new rectangle intersects with the minimal horizontal boundaries of the current depth column.

        Args:
            rect (fitz.Rect): Rect of the depth column entry to be appended.

        Returns:
            bool: True if the new depth column entry can be appended, False otherwise.
        """
        new_middle = (rect.x0 + rect.x1) / 2
        if (self.rect().width < rect.width or self.rect().x0 < new_middle < self.rect().x1) and (
            rect.x0 <= self.min_x1 and self.max_x0 <= rect.x1
        ):
            return True
        return False
