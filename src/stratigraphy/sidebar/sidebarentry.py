"""Contains a dataclass for depth column entries, which indicate the measured depth of an interface between layers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pymupdf

ValueT = TypeVar("ValueT")


@dataclass
class SidebarEntry(abc.ABC, Generic[ValueT]):
    """Abstract class for sidebar entries (e.g. DepthColumnEntry or LayerIdentifierEntry)."""

    rect: pymupdf.Rect
    value: ValueT


@dataclass
class DepthColumnEntry(SidebarEntry[float]):  # noqa: D101
    """Class to represent a depth column entry."""

    rect: pymupdf.Rect
    value: float
    has_decimal_point: bool = False

    def __repr__(self) -> str:
        return str(self.value)

    def to_json(self) -> dict[str, Any]:
        """Convert the depth column entry to a JSON serializable format."""
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "has_decimal_point": self.has_decimal_point,
        }

    @classmethod
    def from_json(cls, data: dict) -> DepthColumnEntry:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the depth column entry.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(rect=pymupdf.Rect(data["rect"]), value=data["value"], has_decimal_point=data["has_decimal_point"])

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
