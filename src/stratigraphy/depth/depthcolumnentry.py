"""Contains a dataclass for depth column entries, which indicate the measured depth of an interface between layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fitz


@dataclass
class DepthColumnEntry:  # noqa: D101
    """Class to represent a depth column entry."""

    rect: fitz.Rect
    value: float | int

    def __repr__(self) -> str:
        return str(self.value)

    def to_json(self) -> dict[str, Any]:
        """Convert the depth column entry to a JSON serializable format."""
        return {"value": self.value, "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1]}

    @classmethod
    def from_json(cls, data: dict) -> DepthColumnEntry:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the depth column entry.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(rect=fitz.Rect(data["rect"]), value=data["value"])
