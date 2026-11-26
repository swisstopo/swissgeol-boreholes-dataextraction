"""Module for clustering DepthColumnEntries when extracting sidebars."""

import dataclasses
import math
from typing import Generic, Self, TypeVar

import numpy as np
import pymupdf
from scipy.stats import gaussian_kde

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
        skew_degrees = cls._estimate_skew(entries)
        skew_slope = math.tan(math.radians(skew_degrees or 0))

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

                # Check if the x-ranges of the two entries overlap after correcting for the estimated page skew:
                # each x-coordinate of the bottom entry is shifted back to the level of the top entry.
                middle1 = (entry1.rect.y0 + entry1.rect.y1) / 2
                middle2 = (entry2.rect.y0 + entry2.rect.y1) / 2
                skew_correction = (middle2 - middle1) * skew_slope
                corrected_x0 = entry2.rect.x0 + skew_correction
                corrected_x1 = entry2.rect.x1 + skew_correction

                if max(entry1.rect.x0, corrected_x0) < min(entry1.rect.x1, corrected_x1):
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

    @staticmethod
    def _estimate_skew(entries: list[EntryT]) -> float | None:
        """Estimate the skew (deviation from vertical) of the entries on the page, in degrees.

        All angles of the lines connecting respectively the left-edges, midpoints and right-edges of the entries are
        computed, keeping only the angles that are within 5 degrees of vertical. Each remaining angle contributes a
        Gaussian to a continuous orientation density function (kernel density estimation). The peak of this density
        function is found with an accuracy of 0.01 degrees. The result is 0 for perfectly vertical entries, positive
        for a clockwise skew and negative for a counter-clockwise skew.

        Args:
            entries: The entries to estimate the skew from.

        Returns:
            The estimated skew in degrees, or None if no near-vertical pairs of entries were found.
        """
        max_skew_degrees = 5

        def midpoint(entry: EntryT) -> Point:
            rect = entry.rect
            return Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

        def left_edge(entry: EntryT) -> Point:
            rect = entry.rect
            return Point(rect.x0, (rect.y0 + rect.y1) / 2)

        def right_edge(entry: EntryT) -> Point:
            rect = entry.rect
            return Point(rect.x1, (rect.y0 + rect.y1) / 2)

        # Collect all near-vertical angles, expressed as deviation from vertical in degrees.
        deviations: list[float] = []
        for point_getter in [left_edge, midpoint, right_edge]:
            points = [point_getter(entry) for entry in entries]
            for index1, point1 in enumerate(points):
                for point2 in points[index1 + 1 :]:
                    if point1 == point2:
                        continue
                    angle = Line(point1, point2).angle
                    deviation = angle - 90 if angle > 0 else angle + 90
                    if abs(deviation) <= max_skew_degrees:
                        deviations.append(deviation)

        # If there is nothing to smooth over (fewer than two samples, or no spread at all), gaussian_kde would fail
        # on the singular covariance matrix. In this degenerate case, the density peaks at the mean of the angles.
        # Note: these guards must run before np.std()/np.mean() to avoid numpy warnings on empty slices.
        if not deviations:
            return None
        if len(deviations) < 2:
            return round(float(np.mean(deviations)), 2)

        # Bandwidth of the Gaussian that each angle contributes to the density function: the standard deviation of
        # the observed angles (Scott's rule would shrink it for large samples, which could produce multiple peaks),
        # with a floor to keep the density stable for very concentrated samples.
        std = float(np.std(deviations))
        if std == 0:
            return round(float(np.mean(deviations)), 2)
        bandwidth = max(0.5, std)

        # Continuous orientation density function: each observed angle contributes a Gaussian (KDE).
        # The bw_method factor will be scaled by the standard deviation of the values, but we actualy want a fixed
        # bandwidth. Therefore, we divide by the std before passing the parameter.
        density = gaussian_kde(deviations, bw_method=bandwidth / float(np.std(deviations, ddof=1)))

        # Find the global peak of the density function, to an accuracy of 0.01 degrees. A local optimizer could
        # converge to a minor peak in a multimodal density, so instead we do an exhaustive grid search. The grid
        # spacing of 0.005 degrees is well below the bandwidth, so the global peak cannot be missed.
        grid = np.arange(-max_skew_degrees, max_skew_degrees + 0.005, 0.005)
        return round(float(grid[np.argmax(density(grid))]), 2)


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
