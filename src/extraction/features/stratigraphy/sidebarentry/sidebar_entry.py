"""Contains classes for different types of sidebar entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import pymupdf

ValueT = TypeVar("ValueT")


# TODO unsafe_hash is necessary for clustering, and because we can't use frozen=True because of the mutable
# relative_shift field of the DepthColumnEntry. Ideally we should separate these concerns more thoroughly in the
# future, so that this unsafe_hash becomes unnecessary.
@dataclass(unsafe_hash=True)
class SidebarEntry(Generic[ValueT]):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    value: ValueT
    rect: pymupdf.Rect
    page_number: int


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""

    pass
