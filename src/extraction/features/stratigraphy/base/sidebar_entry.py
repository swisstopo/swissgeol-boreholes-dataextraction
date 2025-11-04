"""Contains a dataclass for depth column entries, which indicate the measured depth of an interface between layers."""

from __future__ import annotations

import abc
from typing import Generic, TypeVar

import pymupdf

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin

ValueT = TypeVar("ValueT")


class SidebarEntry(abc.ABC, Generic[ValueT], RectWithPageMixin):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    def __init__(self, value: ValueT, rect: pymupdf.rect, page_number: int):
        self.value = value
        self.rect_with_page = RectWithPage(rect, page_number)


class DepthColumnEntry(SidebarEntry[float]):
    """Represents a depth value extracted from the document.

    DepthColumnEntry are used during the extraction process to hold depth data, which will later be part Intervals
    or Sidebars. Unlike `LayerDepthsEntry`, which is used for visualization after extraction, this class is part
    of the core extraction logic, and is the building block for larger object like Sidebars.
    """

    def __init__(self, value: ValueT, rect: pymupdf.rect, page_number: int, has_decimal_point: bool = False):
        super().__init__(value, rect, page_number)
        self.has_decimal_point = has_decimal_point

    def __repr__(self) -> str:
        return str(self.value)

    @classmethod
    def from_string_value(cls, rect: pymupdf.Rect, string_value: str, page_number: int) -> DepthColumnEntry:
        """Creates a DepthColumnEntry from a string representation of the value.

        Args:
            rect (pymupdf.Rect): The rectangle that defines where the entry was found on the PDF page.
            string_value (str): A string representation of the value.
            page_number (int): The page number.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(
            rect=rect,
            value=abs(float(string_value.replace(",", "."))),
            page_number=page_number,
            has_decimal_point="." in string_value,
        )


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""

    pass


class SpulprobeEntry(SidebarEntry[float]):
    """Sidebar entry of type Sp. X m, for boreholes with dicrete sampled depths instead of continued intervals."""

    pass
