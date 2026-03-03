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
        perfect_assignments: dict[EntryT, set[int]] = {entry: set() for entry in entries}
        assignments: dict[EntryT, set[int]] = {entry: set() for entry in entries}

        # iterate over all possibilities for the topmost entry of a cluster
        for index1, entry1 in enumerate(entries):
            midpoint1 = midpoint(entry1)

            # iterate over all possibilities for the bottom entry of a cluster
            for index2, entry2 in enumerate(entries[:index1:-1]):
                index2 = len(entries) - 1 - index2  # use index relative to the full list of entries
                if not perfect_assignments[entry1].isdisjoint(perfect_assignments[entry2]):
                    # skip if the entries already belong to the same cluster
                    continue

                midpoint2 = midpoint(entry2)
                angle = Line(midpoint1, midpoint2).angle

                if abs(abs(angle) - 90) <= max_skew_degrees:
                    cluster_span = ClusterSpan(entry1.rect, entry2.rect)

                    intermediate_entries = []
                    perfect_fits = []
                    not_so_good_fit_count = 0
                    for entry3 in entries[index1 + 1 : index2]:
                        cluster_span_fit = ClusterSpanFit(cluster_span, entry3.rect)
                        if cluster_span_fit.perfect_fit():
                            intermediate_entries.append(entry3)
                            perfect_fits.append(entry3)
                        elif cluster_span_fit.good_fit():
                            intermediate_entries.append(entry3)

                        if cluster_span_fit.not_so_good_fit():
                            not_so_good_fit_count += 1

                    if intermediate_entries:
                        cluster = Cluster([entry1, *intermediate_entries, entry2])

                        if not_so_good_fit_count >= 0.5 * len(intermediate_entries):
                            # if the number of entries with a not-so-good fit is more than half of the number of
                            # intermediate entries -> skip
                            continue

                        if len(set.intersection(*[assignments[entry] for entry in cluster.entries])):
                            # cluster is already fully contained in an existing cluster -> skip
                            continue

                        cluster_index = len(clusters)
                        for entry in [entry1, *perfect_fits, entry2]:
                            perfect_assignments[entry].add(cluster_index)
                        for entry in cluster.entries:
                            assignments[entry].add(cluster_index)
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


@dataclasses.dataclass
class ClusterSpanFit:
    """Class that captures data on how well a rect fits a given cluster span."""

    cluster_span: ClusterSpan
    rect: pymupdf.Rect

    def __post_init__(self):
        avg_y = (self.rect.y0 + self.rect.y1) / 2

        self.x0_expected = self.cluster_span.left_line.x_from_y(avg_y)
        self.x1_expected = self.cluster_span.right_line.x_from_y(avg_y)

    def perfect_fit(self) -> bool:
        return self.x0_expected <= self.rect.x0 and self.rect.x1 <= self.x1_expected

    def good_fit(self) -> bool:
        reference_rect = pymupdf.Rect(self.x0_expected, self.rect.y0, self.x1_expected, self.rect.y1)

        # Accept rects that are fully within the margins and have some minimal overlap
        if (
            self.x0_expected - self.cluster_span.margin < self.rect.x0
            and self.rect.x1 < self.x1_expected + self.cluster_span.margin
            and x_overlap_significant_largest(reference_rect, self.rect, level=0.1)
        ):
            return True

        # Also accept rects that have a significant intersection with the expected location, even if they extend beyond
        # the margins:
        return x_overlap_significant_largest(reference_rect, self.rect, level=0.4)

    def not_so_good_fit(self) -> bool:
        reference_rect = pymupdf.Rect(self.x0_expected, self.rect.y0, self.x1_expected, self.rect.y1)
        minimal_fit = x_overlap_significant_largest(reference_rect, self.rect, level=0.01)
        not_good_fit = not x_overlap_significant_largest(reference_rect, self.rect, level=0.4)
        return minimal_fit and not_good_fit
