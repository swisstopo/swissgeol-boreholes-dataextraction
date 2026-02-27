"""Module for clustering DepthColumnEntries when extracting sidebars."""

import abc
import dataclasses
from collections.abc import Callable
from typing import Generic, Self, TypeVar

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.geometry.util import x_overlap_significant_largest

EntryT = TypeVar("EntryT")


@dataclasses.dataclass
class Cluster(abc.ABC, Generic[EntryT]):
    """Class that groups together values that potentially belong to the same sidebar."""

    entries: list[EntryT]

    @classmethod
    def create_clusters(cls, entries: list[EntryT], entry_to_rect: Callable[[EntryT], pymupdf.Rect]) -> list[Self]:
        def midpoint(entry: EntryT) -> Point:
            rect = entry_to_rect(entry)
            return Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

        max_skew_degrees = 5

        clusters: list[Cluster[EntryT]] = []
        # maps every entry to the set of indices of the clusters that contain this entry
        assignments: dict[EntryT, set[int]] = {entry: set() for entry in entries}

        # iterate over all possibilities for the topmost entry of a cluster
        for index1, entry1 in enumerate(entries):
            midpoint1 = midpoint(entry1)

            # iterate over all possibilities for the bottom entry of a cluster
            for index2, entry2 in enumerate(entries[:index1:-1]):
                index2 = len(entries) - 1 - index2  # use index relative to the full list of entries
                if not assignments[entry1].isdisjoint(assignments[entry2]):
                    # skip if the entries already belong to the same cluster
                    continue

                midpoint2 = midpoint(entry2)
                angle = Line(midpoint1, midpoint2).angle

                if abs(abs(angle) - 90) <= max_skew_degrees:
                    cluster_span = ClusterSpan(entry1.rect, entry2.rect)

                    intermediate_entries = []
                    for entry3 in entries[index1 + 1 : index2]:
                        if cluster_span.good_fit(entry3.rect):
                            intermediate_entries.append(entry3)

                    if intermediate_entries:
                        cluster = Cluster([entry1, *intermediate_entries, entry2])
                        for entry in cluster.entries:
                            assignments[entry].add(len(clusters))
                        clusters.append(cluster)

        return clusters


@dataclasses.dataclass
class ClusterSpan:
    """Class for the first and final entries that can generate a cluster."""

    start_rect: pymupdf.Rect
    end_rect: pymupdf.Rect

    def __post_init__(self):
        self.left_line = Line(
            Point(self.start_rect.x0, (self.start_rect.y0 + self.start_rect.y1) / 2),
            Point(self.end_rect.x0, (self.end_rect.y0 + self.end_rect.y1) / 2),
        )
        self.right_line = Line(
            Point(self.start_rect.x1, (self.start_rect.y0 + self.start_rect.y1) / 2),
            Point(self.end_rect.x1, (self.end_rect.y0 + self.end_rect.y1) / 2),
        )
        # The taller the font, the more flexible we are. We use half the average height of start and end rect.
        self.margin = (self.start_rect.height + self.end_rect.height) / 4

    def good_fit(self, rect: pymupdf.Rect) -> bool:
        avg_y = (rect.y0 + rect.y1) / 2

        x0_expected = self.left_line.x_from_y(avg_y) - self.margin
        x1_expected = self.right_line.x_from_y(avg_y) + self.margin

        # Accept rects that are fully within the margins
        if x0_expected - self.margin < rect.x0 < rect.x1 < x1_expected + self.margin:
            return True

        # Also accept rects that have a significant intersection with the expected location, even if they extend beyond
        # the margins:
        reference_rect = pymupdf.Rect(x0_expected, rect.y0, x1_expected, rect.y1)
        return x_overlap_significant_largest(reference_rect, rect, level=0.4)
