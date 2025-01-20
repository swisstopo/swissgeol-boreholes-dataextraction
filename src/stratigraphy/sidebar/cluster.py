"""Module for clustering DepthColumnEntries when extracting sidebars."""

import abc
import dataclasses
from typing import Generic, TypeVar

import fitz

from ..util.util import x_overlap_significant_largest
from .sidebarentry import SidebarEntry

EntryT = TypeVar("EntryT", bound=SidebarEntry)


@dataclasses.dataclass
class Cluster(abc.ABC, Generic[EntryT]):
    """Class that groups together values that potentially belong to the same sidebar."""

    reference_rect: fitz.Rect
    entries: list[EntryT]

    def append_if_fits_and_return_good_fit(self, entry: EntryT):
        """Appends a new entry to this cluster if it fits. Otherwise, this cluster remains unchanged.

        Args:
            entry: an entry to be added to the cluster, if it fits

        Returns: True if the new entry is a good fit for the cluster, i.e. there is no need to create a new cluster
                 with this element as the reference.
        """
        if x_overlap_significant_largest(self.reference_rect, entry.rect, 0.1):
            self.entries.append(entry)
            return x_overlap_significant_largest(self.reference_rect, entry.rect, 0.75)

        return False
