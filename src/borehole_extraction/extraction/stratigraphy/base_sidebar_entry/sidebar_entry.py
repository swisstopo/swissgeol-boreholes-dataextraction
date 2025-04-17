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

    value: ValueT
    rect: pymupdf.Rect


@dataclass
class DepthColumnEntry(SidebarEntry[float]):
    """Class to represent a depth column entry."""

    value: float
    rect: pymupdf.Rect
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

    @classmethod
    def from_json(cls, data: dict) -> DepthColumnEntry:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths entry.

        Returns:
            DepthColumnEntry: the corresponding LayerDepthsEntry object.
        """
        return cls(value=data["value"], rect=pymupdf.Rect(data["rect"]))

    def to_json(self):
        """Convert the LayerDepthsEntry object to a JSON serializable format."""
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
        }


class LayerIdentifierEntry(SidebarEntry[str]):
    """Class for a layer identifier entry."""
