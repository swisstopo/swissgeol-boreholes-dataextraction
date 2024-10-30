"""Definition of the DepthsMaterialsColumnPairs class."""

from dataclasses import dataclass

import fitz
from stratigraphy.depthcolumn.depthcolumn import DepthColumn, DepthColumnFactory


@dataclass
class DepthsMaterialsColumnPairs:
    """A class to represent pairs of depth columns and material descriptions."""

    depth_column: DepthColumn | None
    material_description_rect: fitz.Rect
    page: int

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return (
            f"DepthsMaterialsColumnPairs(depth_column={self.depth_column},"
            f"material_description_rect={self.material_description_rect}, page={self.page})"
        )

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "depth_column": self.depth_column.to_json() if self.depth_column else None,
            "material_description_rect": [
                self.material_description_rect.x0,
                self.material_description_rect.y0,
                self.material_description_rect.x1,
                self.material_description_rect.y1,
            ],
            "page": self.page,
        }

    @classmethod
    def from_json(cls, json_depths_materials_column_pairs: dict) -> "DepthsMaterialsColumnPairs":
        """Converts a dictionary to an object.

        Args:
            json_depths_materials_column_pairs (dict): A dictionary representing the depths materials column pairs.

        Returns:
            DepthsMaterialsColumnPairs: The depths materials column pairs object.
        """
        depth_column_entry = json_depths_materials_column_pairs["depth_column"]
        depth_column = DepthColumnFactory.create(depth_column_entry) if depth_column_entry else None
        material_description_rect = fitz.Rect(json_depths_materials_column_pairs["material_description_rect"])
        page = json_depths_materials_column_pairs["page"]

        return cls(depth_column, material_description_rect, page)
