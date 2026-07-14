"""Contains classes for different types of sidebar entries."""

from __future__ import annotations

import abc
from typing import Generic, TypeVar

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin

ValueT = TypeVar("ValueT")


class SidebarEntry(abc.ABC, Generic[ValueT], RectWithPageMixin):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    def __init__(self, value: ValueT, rect: pymupdf.Rect, page_number: int):
        self.value = value
        self.rect_with_page = RectWithPage(rect, page_number)


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""

    pass
