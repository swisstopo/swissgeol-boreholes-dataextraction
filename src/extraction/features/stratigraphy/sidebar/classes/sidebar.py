"""This module contains the Sidebar class, used to represent a depth column (or similar) of a borehole profile."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import SidebarEntry
from extraction.features.stratigraphy.interval.interval import IntervalBlockPair, IntervalZone
from swissgeol_doc_processing.geometry.util import x_overlap_significant_smallest
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_affinity import Affinity
from swissgeol_doc_processing.utils.file_utils import read_params

EntryT = TypeVar("EntryT", bound=SidebarEntry)

matching_params = read_params("matching_params.yml")


@dataclass
class Sidebar(abc.ABC, Generic[EntryT]):
    """Abstract base class for representing depths or other data displayed to the side of material descriptions.

    A `Sidebar` holds a list of structured entries (e.g., depth markers, labels, intervals) that were
    identified during the extraction phase.

    Sidebars are used strictly during extraction. They come in various forms, such as depth columns,
    AtoB intervals, or labeled layer identifiers, depending on how the source document expresses stratigraphy.
    Once validated and processed, sidebar content is transformed into a finalized `ExtractedBorehole` where
    each entry is mapped to a `Layer` object, which are used for visualization, evaluation, or export.
    """

    entries: list[EntryT]
    skipped_entries: list[EntryT] = field(default_factory=list)
    kind: ClassVar[str] = "base_sidebar"

    @property
    def all_entries(self):
        return self.entries + self.skipped_entries

    def rects(self) -> list[pymupdf.Rect]:
        """Get the rectangles of the depth column entries."""
        return [entry.rect for entry in self.entries]

    @property
    def rect(self) -> pymupdf.Rect:
        """Get the bounding box of the depth column entries."""
        x0 = min([rect.x0 for rect in self.rects()])
        x1 = max([rect.x1 for rect in self.rects()])
        y0 = min([rect.y0 for rect in self.rects()])
        y1 = max([rect.y1 for rect in self.rects()])
        return pymupdf.Rect(x0, y0, x1, y1)

    @property
    def max_x0(self) -> float:
        """Get the maximum x0 value of the depth column entries."""
        return max([rect.x0 for rect in self.rects()])

    @property
    def min_x1(self) -> float:
        """Get the minimum x1 value of the depth column entries."""
        return min([rect.x1 for rect in self.rects()])

    @abc.abstractmethod
    def get_interval_zone(self) -> list[IntervalZone]:
        """Get the interval zones defined by the sidebar entries."""
        pass

    @staticmethod
    def get_zones_from_entries(entries: list[EntryT], include_open_ended: bool = True):
        zones = [
            IntervalZone(entry.rect, next_entry.rect, entry)
            for entry, next_entry in zip(entries, entries[1:], strict=False)
        ]
        if include_open_ended:
            return zones + [IntervalZone(entries[-1].rect, None, entries[-1])]
        return zones

    @staticmethod
    @abc.abstractmethod
    def dp_scoring_fn(interval_zone: IntervalZone, line: TextLine) -> float:
        """Scoring function for dynamic programming matching of description lines to interval zones."""
        pass

    def default_score(interval_zone: IntervalZone, line: TextLine) -> float:
        """Returns a default score of 1.0 if the text line is within the zone, or 0.0 otherwise.

        A text line is considered inside the zone if its middle y-coordinate lies between the top boundaries of the
        start and end entries.

        Args:
            interval_zone (IntervalZone): The interval zone to score against.
            line (TextLine): The text line to score.

        Returns:
            float: The score for the given interval zone and text line.
        """
        start_top = interval_zone.start.y0 if interval_zone.start else None
        end_top = interval_zone.end.y0 if interval_zone.end else None
        line_mid = (line.rect.y0 + line.rect.y1) / 2
        if (start_top is None or line_mid > start_top) and (end_top is None or line_mid < end_top):
            return 1.0  # textline is inside the depth interval
        return 0.0

    def dp_weighted_affinities(self, affinities: list[Affinity]) -> list[float]:
        """Returns the weighted affinity used for dynamic programming, with the weights specific to each sidebar type.

        Args:
            affinities (list[Affinity]): the list containing all Affinty of each description line with the one above.

        Returns:
            list[float]: A list of floats condensing the affinities, according to the sidebar type.
        """
        affinity_params = matching_params["affinity_params"][self.kind]
        affinity_importance = affinity_params["importance"]
        weights = affinity_params["weights"]
        return [affinity_importance * affinity.weighted_mean_affinity(**weights) for affinity in affinities]

    def post_processing(
        self, interval_lines_mapping: list[tuple[IntervalZone, list[TextLine]]]
    ) -> list[IntervalBlockPair]:
        """Post-process the matched interval zones and description lines into IntervalBlockPairs."""
        return [IntervalBlockPair(zone.related_interval, TextBlock(lines)) for zone, lines in interval_lines_mapping]


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


@dataclass
class SidebarQualityMetrics:
    """Quality metrics for sidebar extraction on a page.

    Attributes:
        number_of_good_sidebars: Number of sidebars that passed quality filters.
        best_sidebar_score: The highest matching score among all sidebars.
    """

    number_of_good_sidebars: int
    best_sidebar_score: float


def noise_count(sidebar: Sidebar, line_rtree: fastquadtree.RectQuadTreeObjects) -> int:
    """Counts the number of text lines that intersect with the Sidebar entries.

    Args:
        sidebar (Sidebar): Sidebar object for which the noise count is calculated.
        line_rtree (fastquadtree.RectQuadTreeObjects): Pre-built R-tree of all text lines on page for spatial queries.

    Returns:
        int: The number of text lines that intersect with the Sidebar entries but are not part of it.
    """
    sidebar_rect = sidebar.rect
    intersecting_lines = _get_intersecting_lines(line_rtree, sidebar_rect)

    def significant_intersection(line: TextLine) -> bool:
        line_rect = line.rect
        x_overlap = x_overlap_significant_smallest(sidebar_rect, line_rect, 0.25)
        intersects = line_rect.intersects(sidebar_rect)
        return x_overlap and intersects

    def not_in_entries(line: TextLine) -> bool:
        line_rect = line.rect
        return not any(line_rect.intersects(entry.rect) for entry in sidebar.all_entries)

    return sum(1 for line in intersecting_lines if significant_intersection(line) and not_in_entries(line))


def _get_intersecting_lines(line_rtree: fastquadtree.RectQuadTreeObjects, rect: pymupdf.Rect) -> list[TextLine]:
    """Retrieve all words from the page intersecting with Sidebar bounding box."""
    intersecting_lines = [item.obj for item in line_rtree.query((rect.x0, rect.y0, rect.x1, rect.y1))]
    return [line for line in intersecting_lines if any(char.isalnum() for char in line.text)]
