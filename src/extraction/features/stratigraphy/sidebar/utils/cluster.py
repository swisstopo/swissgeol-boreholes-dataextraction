"""Module for clustering DepthColumnEntries when extracting sidebars."""

import dataclasses
from typing import Generic, Self, TypeVar

import pymupdf

from extraction.features.stratigraphy.sidebarentry.sidebar_entry import SidebarEntry
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.geometry.util import x_overlap_significant_largest, x_overlap_significant_smallest
from swissgeol_doc_processing.utils.table_detection import TableStructure

EntryT = TypeVar("EntryT", bound=SidebarEntry)


@dataclasses.dataclass
class VerticalLinePartition(Generic[EntryT]):
    """Represents how a vertical line partitions depth entries."""

    line: Line
    left: set[EntryT]
    right: set[EntryT]
    left_extended: set[EntryT]
    right_extended: set[EntryT]

    @classmethod
    def from_line(cls, line: Line, entries: list[EntryT]) -> "VerticalLinePartition[EntryT]":
        left = set()
        right = set()
        left_extended = set()
        right_extended = set()
        line_y0 = min(line.start.y, line.end.y)
        line_y1 = max(line.start.y, line.end.y)
        for entry in entries:
            rect = entry.rect

            # Only consider entries that are to the right or left of the extension of the line that goes one line
            # length below and one line length above the actual line.
            is_inside_extended = line_y0 - line.length <= rect.y1 and rect.y0 <= line_y1 + line.length
            if is_inside_extended:
                entry_middle = (rect.top_left + rect.bottom_right) / 2
                if entry_middle.x < line.x_from_y(entry_middle.y):
                    inside_set, extended_set = left, left_extended
                else:
                    inside_set, extended_set = right, right_extended

                extended_set.add(entry)
                if line_y0 <= rect.y1 and rect.y0 <= line_y1:
                    inside_set.add(entry)

        return VerticalLinePartition(
            line, left=left, right=right, left_extended=left_extended, right_extended=right_extended
        )

    def no_conflict(self, partition: "VerticalLinePartition[EntryT]") -> bool:
        return partition.left.isdisjoint(self.right) and partition.right.isdisjoint(self.left)

    def splits(self, entries: list[EntryT]) -> bool:
        return (not self.left_extended.isdisjoint(entries) and not self.right.isdisjoint(entries)) or (
            not self.left.isdisjoint(entries) and not self.right_extended.isdisjoint(entries)
        )


@dataclasses.dataclass
class Cluster(Generic[EntryT]):
    """Class that groups together values that potentially belong to the same sidebar."""

    entries: list[EntryT]

    @classmethod
    def create_clusters(
        cls,
        entries: list[EntryT],
        table_structure: TableStructure | None,
        allow_size_two: bool = False,
    ) -> list[Self]:
        def midpoint(entry: EntryT) -> Point:
            rect = entry.rect
            return Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

        def left_edge(entry: EntryT) -> Point:
            rect = entry.rect
            return Point(rect.x0, (rect.y0 + rect.y1) / 2)

        def right_edge(entry: EntryT) -> Point:
            rect = entry.rect
            return Point(rect.x1, (rect.y0 + rect.y1) / 2)

        max_skew_degrees = 5

        clusters: list[Cluster[EntryT]] = []
        # maps every entry to the set of indices of the clusters that contain this entry
        perfect_assignments: dict[EntryT, set[int]] = {entry: set() for entry in entries}
        assignments: dict[EntryT, set[int]] = {entry: set() for entry in entries}

        vertical_partitions = []
        if table_structure is not None:
            vertical_partitions = [
                VerticalLinePartition.from_line(line, entries) for line in table_structure.vertical_lines
            ]

        # iterate over all possibilities for the topmost entry of a cluster
        for index1, entry1 in enumerate(entries):
            # iterate over all possibilities for the bottom entry of a cluster
            for index2, entry2 in enumerate(entries[:index1:-1]):
                index2 = len(entries) - 1 - index2  # use index relative to the full list of entries
                if not perfect_assignments[entry1].isdisjoint(perfect_assignments[entry2]):
                    # skip if the entries already belong to the same cluster
                    continue

                # check if rects align (left edge, right edge or midpoint)
                accepted = False
                for point_getter in (midpoint, left_edge, right_edge):
                    point1 = point_getter(entry1)
                    point2 = point_getter(entry2)
                    angle = Line(point1, point2).angle

                    if abs(abs(angle) - 90) <= max_skew_degrees:
                        accepted = True
                        break

                if accepted:
                    cluster_span = ClusterSpan(entry1.rect, entry2.rect)

                    intermediate_entries = []
                    perfect_fits = []
                    for entry3 in entries[index1 + 1 : index2]:
                        cluster_span_fit = ClusterSpanFit(cluster_span, entry3.rect)
                        if cluster_span_fit.perfect_fit():
                            intermediate_entries.append(entry3)
                            perfect_fits.append(entry3)
                        elif cluster_span_fit.good_fit():
                            intermediate_entries.append(entry3)

                    if allow_size_two or intermediate_entries:
                        cluster = Cluster([entry1, *intermediate_entries, entry2])

                        if len(set.intersection(*[assignments[entry] for entry in cluster.entries])):
                            # cluster is already fully contained in an existing cluster -> skip
                            continue

                        if any(partition.splits(cluster.entries) for partition in vertical_partitions):
                            # there is a vertical line that splits the cluster entries -> skip
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
        if self.x0_expected is None or self.x1_expected is None:
            return False

        return self.x0_expected <= self.rect.x0 and self.rect.x1 <= self.x1_expected

    def good_fit(self) -> bool:
        if self.x0_expected is None or self.x1_expected is None:
            return False

        reference_rect = pymupdf.Rect(self.x0_expected, self.rect.y0, self.x1_expected, self.rect.y1)

        # Accept rects that are fully within the margins and have some minimal overlap
        if (
            self.x0_expected - self.cluster_span.margin < self.rect.x0
            and self.rect.x1 < self.x1_expected + self.cluster_span.margin
            and x_overlap_significant_largest(reference_rect, self.rect, level=0.01)
        ):
            return True

        # Also accept rects that have a significant intersection with the expected location, even if they extend beyond
        # the margins:
        return x_overlap_significant_largest(reference_rect, self.rect, level=0.2)

    def significantly_outside(self) -> bool:
        if self.x0_expected is None or self.x1_expected is None:
            return False

        reference_rect = pymupdf.Rect(self.x0_expected, self.rect.y0, self.x1_expected, self.rect.y1)
        return not x_overlap_significant_smallest(reference_rect, self.rect, level=0.4)
