"""Module for clustering DepthColumnEntries when extracting sidebars."""

import abc
import dataclasses
from collections.abc import Callable
from typing import Generic, Self, TypeVar

import pymupdf
from borehole_extraction.extraction.util_extraction.geometry.util import x_overlap_significant_largest

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
