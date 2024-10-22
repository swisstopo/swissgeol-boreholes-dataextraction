"""Contains dataclasses for entries in a depth column."""

from typing import Any

import fitz


class DepthColumnEntry:  # noqa: D101
    """Class to represent a depth column entry."""

    def __init__(self, rect: fitz.Rect, value: float, page_number: int):
        self.rect = rect
        self.value = value
        self.page_number = page_number

    def __repr__(self) -> str:
        return str(self.value)

    def to_json(self) -> dict[str, Any]:
        """Convert the depth column entry to a JSON serializable format."""
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.page_number,
        }

    @classmethod
    def from_json(cls, json_depth_column_entry: dict) -> "DepthColumnEntry":
        """Converts a dictionary to an object.

        Args:
            json_depth_column_entry (dict): A dictionary representing the depth column entry.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        return cls(
            rect=fitz.Rect(json_depth_column_entry["rect"]),
            value=json_depth_column_entry["value"],
            page_number=json_depth_column_entry["page"],
        )


class AnnotatedDepthColumnEntry(DepthColumnEntry):  # noqa: D101
    """Class to represent a depth column entry obtained from LabelStudio.

    The annotation process in label studio does not come with rectangles for depth column entries.
    Therefore, we set them to None.
    """

    def __init__(self, value):
        super().__init__(None, value, None)

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "rect": self.rect,
            "page": self.page_number,
        }


class LayerDepthColumnEntry:  # noqa: D101
    """Class to represent a layer depth column entry."""

    def __init__(self, start: DepthColumnEntry, end: DepthColumnEntry):
        self.start = start
        self.end = end

        assert start.page_number == end.page_number, "Start and end entries are on different pages."

    def __repr__(self) -> str:
        return f"{self.start.value}-{self.end.value}"

    @property
    def rect(self) -> fitz.Rect:
        """Get the rectangle of the layer depth column entry."""
        return fitz.Rect(self.start.rect).include_rect(self.end.rect)

    def to_json(self) -> dict[str, Any]:
        """Convert the layer depth column entry to a JSON serializable format."""
        return {
            "start": self.start.to_json(),
            "end": self.end.to_json(),
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.start.page_number,
        }

    @classmethod
    def from_json(cls, json_layer_depth_column_entry: dict) -> "LayerDepthColumnEntry":
        """Converts a dictionary to an object.

        Args:
            json_layer_depth_column_entry (dict): A dictionary representing the layer depth column entry.

        Returns:
            LayerDepthColumnEntry: The layer depth column entry object.
        """
        start = DepthColumnEntry.from_json(json_layer_depth_column_entry["start"])
        end = DepthColumnEntry.from_json(json_layer_depth_column_entry["end"])
        return cls(start, end)
