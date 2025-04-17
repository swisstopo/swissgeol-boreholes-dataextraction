"""Classes for JSON-serializable bounding boxes of different parts of a borehole profile."""

import math
from dataclasses import dataclass

import pymupdf
from borehole_extraction.extraction.stratigraphy.sidebar_classes.sidebar import Sidebar
from borehole_extraction.extraction.util_extraction.geometry.geometry_dataclasses import BoundingBox


@dataclass
class MaterialDescriptionRectWithSidebar:
    """A class to represent pairs of sidebar and material description rectangle."""

    sidebar: Sidebar | None
    material_description_rect: pymupdf.Rect
    noise_count: int = 0

    @property
    def score_match(self) -> float:
        """Scores the match between a sidebar and a material description.

        For pairs that have a sidebar, the score is
        - positively influenced by the width of the material description bounding box
        - negatively influenced by the horizontal distance between (the right-hand-side of) the sidebar and (the
          left-hand-side of) the material descriptions
        - positively influenced by the height of the sidebar
        - negatively influenced by vertical distance between the top of the sidebar and the top of the material
          descriptions, and the vertical distance between the bottom of the sidebar and the bottom of the material
          descriptions
        The resulting score is also reduced if the sidebar has a high noise count (many unrelated tokens in between
        the extracted depths values).

        Pairs without a sidebar receive a default score of 0.

        Returns:
            float: The score of the match. Better matches have a higher score value.
        """
        if self.sidebar:
            rect = self.sidebar.rect()
            top = rect.y0
            bottom = rect.y1
            right = rect.x1
            x_distance = abs(right - self.material_description_rect.x0)
            y_distance = abs(top - self.material_description_rect.y0) + abs(bottom - self.material_description_rect.y1)

            height = bottom - top

            return (self.material_description_rect.width - x_distance + height - 2 * y_distance) * math.pow(
                0.8, 10 * self.noise_count / len(self.sidebar.entries)
            )
        else:
            return 0


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

    def get_outer_rect(self) -> pymupdf.Rect:
        """Returns the extreme bounding rectangle.

        Computes the smallest rectangle that encloses all bounding boxes in this PageBoundingBoxes object.

        Returns:
            pymupdf.Rect: The bounding rectangle.
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

        return pymupdf.Rect(min_x, min_y, max_x, max_y)
