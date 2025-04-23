"""Contains a dataclass for depth column entries, which indicate the measured depth of an interface between layers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

import pymupdf

ValueT = TypeVar("ValueT")


@dataclass
class SidebarEntry(abc.ABC, Generic[ValueT]):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    rect: pymupdf.Rect
    value: ValueT


@dataclass
class DepthColumnEntry(SidebarEntry[float]):
    """Represents a depth value extracted from the document.

    DepthColumnEntry are used during the extraction process to hold depth data, which will later be part Intervals
    or Sidebars. Unlike `LayerDepthsEntry`, which is used for visualization after extraction, this class is part
    of the core extraction logic, and is the building block for larger object like Sidebars.
    """

    rect: pymupdf.Rect
    value: float
    has_decimal_point: bool = False

    def __repr__(self) -> str:
        return str(self.value)

    @classmethod
    def from_string_value(cls, rect: pymupdf.Rect, string_value: str) -> DepthColumnEntry:
        """Creates a DepthColumnEntry from a string representation of the value.

        Args:
            rect (pymupdf.Rect): The rectangle that defines where the entry was found on the PDF page.
            string_value (str): A string representation of the value.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(rect=rect, value=abs(float(string_value)), has_decimal_point="." in string_value)


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""
