"""Classes for JSON-serializable bounding boxes of different parts of a borehole profile."""

from dataclasses import dataclass

import fitz
from stratigraphy.depths_materials_column_pairs.material_description_rect_with_sidebar import (
    MaterialDescriptionRectWithSidebar,
)


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

    @classmethod
    def from_json(cls, data) -> "BoundingBox":
        return cls(rect=fitz.Rect(data))


@dataclass
class PageBoundingBoxes:
    """A class to represent the bounding boxes of sidebars and associated material descriptions."""

    sidebar_bbox: BoundingBox | None
    depth_column_entry_bboxes: list[BoundingBox]
    material_description_bbox: BoundingBox
    page: int

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "sidebar_rect": self.sidebar_bbox.to_json() if self.sidebar_bbox else None,
            "depth_column_entries": [entry.to_json() for entry in self.depth_column_entry_bboxes],
            "material_description_rect": self.material_description_bbox.to_json(),
            "page": self.page,
        }

    @classmethod
    def from_json(cls, data) -> "PageBoundingBoxes":
        """Convert a JSON data structure to a BoundingBoxes object."""
        return cls(
            sidebar_bbox=BoundingBox.from_json(data["sidebar_rect"])
            if "sidebar_rect" in data and data["sidebar_rect"]
            else None,
            depth_column_entry_bboxes=[BoundingBox.from_json(entry) for entry in data["depth_column_entries"]],
            material_description_bbox=BoundingBox.from_json(data["material_description_rect"]),
            page=data["page"],
        )

    @classmethod
    def from_material_description_rect_with_sidebar(
        cls, pair: MaterialDescriptionRectWithSidebar, page_number: int
    ) -> "PageBoundingBoxes":
        """Convert a MaterialDescriptionRectWithSidebar instance to a BoundingBoxes object."""
        if pair.sidebar:
            depth_column_bbox = BoundingBox(pair.sidebar.rect())
            depth_column_entry_bboxes = [BoundingBox(entry.rect) for entry in pair.sidebar.entries]
        else:
            depth_column_bbox = None
            depth_column_entry_bboxes = []
        return PageBoundingBoxes(
            sidebar_bbox=depth_column_bbox,
            depth_column_entry_bboxes=depth_column_entry_bboxes,
            material_description_bbox=BoundingBox(pair.material_description_rect),
            page=page_number,
        )

    def get_outer_rect(self) -> fitz.Rect:
        """Returns the extreme bounding rectangle.

        Computes the smallest rectangle that encloses all bounding boxes in this PageBoundingBoxes object.

        Returns:
            fitz.Rect: The bounding rectangle.
        """
        all_bboxes = [self.material_description_bbox] + self.depth_column_entry_bboxes
        if self.sidebar_bbox:
            all_bboxes.append(self.sidebar_bbox)

        if not all_bboxes:
            raise ValueError("No bounding boxes available to determine extreme coordinates.")

        # Compute extreme coordinates
        min_x = min(bbox.rect.x0 for bbox in all_bboxes)
        min_y = min(bbox.rect.y0 for bbox in all_bboxes)
        max_x = max(bbox.rect.x1 for bbox in all_bboxes)
        max_y = max(bbox.rect.y1 for bbox in all_bboxes)

        return fitz.Rect(min_x, min_y, max_x, max_y)
