"""This module contains the Sidebar class, used to represent a depth column (or similar) of a borehole profile."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

import fitz
import rtree

from stratigraphy.lines.line import TextLine
from stratigraphy.sidebar.interval_block_group import IntervalBlockGroup
from stratigraphy.sidebar.sidebarentry import DepthColumnEntry
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.util import x_overlap_significant_smallest

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


@dataclass
class SidebarNoise(Generic[EntryT]):
    """Wrapper class for Sidebar to calculate noise count using intersecting words."""

    sidebar: Sidebar[EntryT]
    noise_count: int

    def __post_init__(self):
        if not isinstance(self.sidebar, Sidebar):
            raise TypeError(f"Expected a Sidebar instance, got {type(self.sidebar).__name__}")

    def __repr__(self):
        return f"SidebarNoise(sidebar={repr(self.sidebar)}, noise_count={self.noise_count})"


def noise_count(sidebar: Sidebar, line_rtree: rtree.index.Index) -> int:
    """Counts the number of text lines that intersect with the Sidebar entries.

    Args:
        sidebar (Sidebar): Sidebar object for which the noise count is calculated.
        line_rtree (rtree.index.Index): Pre-built R-tree of all text lines on page for spatial queries.

    Returns:
        int: The number of text lines that intersect with the Sidebar entries but are not part of it.
    """
    sidebar_rect = sidebar.rect()
    intersecting_lines = _get_intersecting_lines(line_rtree, sidebar_rect)

    def significant_intersection(line: TextLine) -> bool:
        line_rect = line.rect
        x_overlap = x_overlap_significant_smallest(sidebar_rect, line_rect, 0.25)
        intersects = line_rect.intersects(sidebar_rect)
        return x_overlap and intersects

    def not_in_entries(line: TextLine) -> bool:
        line_rect = line.rect
        return not any(line_rect.intersects(entry.rect) for entry in sidebar.entries)

    return sum(1 for line in intersecting_lines if significant_intersection(line) and not_in_entries(line))


def _get_intersecting_lines(line_rtree: rtree.index.Index, rect: fitz.Rect) -> list[TextLine]:
    """Retrieve all words from the page intersecting with Sidebar bounding box."""
    intersecting_lines = list(line_rtree.intersection((rect.x0, rect.y0, rect.x1, rect.y1), objects="raw"))
    return [line for line in intersecting_lines if any(char.isalnum() for char in line.text)]
