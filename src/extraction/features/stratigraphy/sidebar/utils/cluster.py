"""Module for clustering DepthColumnEntries when extracting sidebars."""

import abc
import dataclasses
from collections.abc import Callable
from typing import Generic, Self, TypeVar

import pymupdf
from extraction.features.utils.geometry.util import (
    compute_outer_rect,
    x_distance_with_y_constraint,
    x_overlap_significant_largest,
)

EntryT = TypeVar("EntryT")


@dataclasses.dataclass
class Cluster(abc.ABC, Generic[EntryT]):
    """Class that groups together values that potentially belong to the same sidebar."""

    reference_rect: pymupdf.Rect
    entries: list[EntryT]
    entry_to_rect: Callable[[EntryT], pymupdf.Rect]

    def good_fit(self, entry: EntryT, threshold: float) -> bool:
        return x_overlap_significant_largest(self.reference_rect, self.entry_to_rect(entry), threshold)

    @classmethod
    def create_clusters(cls, entries: list[EntryT], entry_to_rect: Callable[[EntryT], pymupdf.Rect]) -> list[Self]:
        clusters: list[Cluster[EntryT]] = []
        for entry in entries:
            create_new_cluster = True
            for cluster in clusters:
                if cluster.good_fit(entry, 0.1):
                    cluster.entries.append(entry)
                    if cluster.good_fit(entry, 0.75):
                        # If the fit is good (>0.1) but not very good (<0.75), then we both add the element to this
                        # cluster, as well as potentially creating a new cluster starting with this entry. Only if we
                        # have an excellent fit (>0.75) with some cluster, then we completely skip creating a new
                        # cluster.
                        create_new_cluster = False

            if create_new_cluster:
                clusters.append(Cluster(entry_to_rect(entry), [entry], entry_to_rect))

        return clusters

    @classmethod
    def merge_close_clusters(cls, clusters: list["Cluster[EntryT]"]) -> list["Cluster[EntryT]"]:
        """Iteratively merge clusters that are close to each other.

        Args:
            clusters: A list of Cluster objects to merge.

        Returns:
            A list of merged Cluster objects.
        """
        current_clusters = clusters.copy()  # Make a copy to avoid mutating the input

        while True:
            # Flag to track if any merges occurred in this pass
            merged_occurred = False
            new_clusters = []

            while current_clusters:
                base_cluster = current_clusters.pop(0)  # take first cluster
                entries = base_cluster.entries.copy()
                rect = compute_outer_rect(entries)

                # Check remaining clusters for potential merges
                remains = []
                for other in current_clusters:
                    other_rect = compute_outer_rect(other.entries)
                    dist = x_distance_with_y_constraint(rect, other_rect, 0.6)
                    if dist is not None and dist <= 100.0:
                        # Merge with base cluster
                        entries += other.entries
                        rect = compute_outer_rect(entries)
                        merged_occurred = True
                    else:
                        remains.append(other)

                # Add the (potentially) merged cluster to new list
                new_clusters.append(cls(rect, entries, base_cluster.entry_to_rect))
                # Continue with remaining unmerged clusters
                current_clusters = remains

            # If no merges happened this pass, we're done
            if not merged_occurred:
                return new_clusters

            # Otherwise, do another pass with the new clusters
            current_clusters = new_clusters
