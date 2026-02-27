"""Module for clustering DepthColumnEntries when extracting sidebars."""

import abc
import dataclasses
import math
from collections.abc import Callable
from typing import Generic, Self, TypeVar

import pymupdf

from swissgeol_doc_processing.geometry.geometric_line_utilities import _odr_regression
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.geometry.util import x_overlap_significant_largest

EntryT = TypeVar("EntryT")


@dataclasses.dataclass
class Cluster(abc.ABC, Generic[EntryT]):
    """Class that groups together values that potentially belong to the same sidebar."""

    entries: list[EntryT]

    def good_fit(self, entry: EntryT, entry_to_rect: Callable[[EntryT], pymupdf.Rect], threshold: float) -> bool:
        if len(self.entries) == 0:
            return False

        entry_rect = entry_to_rect(entry)
        if len(self.entries) == 1:
            reference_rect = entry_to_rect(self.entries[0])
            return x_overlap_significant_largest(reference_rect, entry_rect, threshold)

        y = [(entry_to_rect(entry).y0 + entry_to_rect(entry).y1) / 2 for entry in self.entries]
        avg_width = sum(entry_to_rect(entry).width for entry in self.entries) / len(self.entries)
        new_y = (entry_to_rect(entry).y0 + entry_to_rect(entry).y1) / 2
        tests = [
            # linear regression on the leftmost edge, works well for left-aligned values
            (lambda rect: rect.x0, lambda x: x, lambda x: x + avg_width),
            # linear regression on the rightmost edge, works well for right-aligned values
            (lambda rect: rect.x1, lambda x: x - avg_width, lambda x: x),
            # linear regression on the midpoint, works well for centered values
            (lambda rect: (rect.x0 + rect.x1) / 2, lambda x: x - avg_width / 2, lambda x: x + avg_width / 2),
        ]
        for get_x, new_x0, new_x1 in tests:
            x = [get_x(entry_to_rect(entry)) for entry in self.entries]
            phi, r = _odr_regression(x, y)
            if math.cos(phi) == 0:
                continue
            # Line equation is `x cos(phi) + y sin(phi) - r = 0`
            # Therefore `x = (r - y sin(phi)) / cos(phi)`
            inferred_x = (r - new_y * math.sin(phi)) / math.cos(phi)
            reference_rect = pymupdf.Rect(new_x0(inferred_x), 0, new_x1(inferred_x), 0)  # y does not matter
            if x_overlap_significant_largest(reference_rect, entry_rect, threshold):
                # as soon as it's a good fit for one of the tests, we return True
                return True

        return False

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
                    cluster = Cluster([entry1, entry2])

                    intermediate_entries = []
                    for entry3 in entries[index1 + 1 : index2]:
                        if cluster.good_fit(entry3, entry_to_rect, 0.7):
                            intermediate_entries.append(entry3)

                    if intermediate_entries:
                        cluster = Cluster([entry1, *intermediate_entries, entry2])
                        for entry in cluster.entries:
                            assignments[entry].add(len(clusters))
                        clusters.append(cluster)

        return clusters
