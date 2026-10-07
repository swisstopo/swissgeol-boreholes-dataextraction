"""Classes for JSON-serializable bounding boxes of different parts of a borehole profile."""

from dataclasses import dataclass

import pymupdf

from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from swissgeol_doc_processing.geometry.geometry_dataclasses import BoundingBox


@dataclass
class PageBoundingBoxes:
    """A class to represent the bounding boxes of sidebars and associated material descriptions."""

    sidebar_bbox: BoundingBox | None
    depth_column_entry_bboxes: list[BoundingBox]
    material_description_bbox: BoundingBox
    page: int

    @classmethod
    def from_sidebar_and_rect(
        cls, sidebar: Sidebar | None, material_description_rect: pymupdf.Rect, page_number: int
    ) -> "PageBoundingBoxes":
        """Convert an optional sidebar and a material description bounding box to a BoundingBoxes object."""
        if sidebar:
            depth_column_bbox = BoundingBox(sidebar.rect)
            depth_column_entry_bboxes = [BoundingBox(entry.rect) for entry in sidebar.entries]
        else:
            depth_column_bbox = None
            depth_column_entry_bboxes = []
        return PageBoundingBoxes(
            sidebar_bbox=depth_column_bbox,
            depth_column_entry_bboxes=depth_column_entry_bboxes,
            material_description_bbox=BoundingBox(material_description_rect),
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
