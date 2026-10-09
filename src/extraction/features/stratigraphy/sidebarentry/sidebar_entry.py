"""Contains classes for different types of sidebar entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import pymupdf

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class SidebarEntry(Generic[ValueT]):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    value: ValueT
    rect: pymupdf.Rect
    page_number: int


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""

    pass
