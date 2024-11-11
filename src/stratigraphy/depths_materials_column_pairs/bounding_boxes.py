"""Definition of the DepthsMaterialsColumnPairs class."""

from dataclasses import dataclass

import fitz


@dataclass
class BoundingBox:
    """A single bounding box, JSON serializable."""

    rect: fitz.Rect

    def to_json(self) -> list[int]:
        """Converts the object to a dictionary.

        Returns:
            list[int]: The object as a list.
        """
        return [
            self.rect.x0,
            self.rect.y0,
            self.rect.x1,
            self.rect.y1,
        ]


@dataclass
class BoundingBoxes:
    """A class to represent the bounding boxes of depth columns and associated material descriptions."""

    depth_column_bbox: BoundingBox | None
    depth_column_entry_bboxes: list[BoundingBox]
    material_description_bbox: BoundingBox
    page: int

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "depth_column_rect": self.depth_column_bbox.to_json() if self.depth_column_bbox else None,
            "depth_column_entries": [entry.to_json for entry in self.depth_column_entry_bboxes],
            "material_description_rect": self.material_description_bbox.to_json(),
            "page": self.page,
        }
