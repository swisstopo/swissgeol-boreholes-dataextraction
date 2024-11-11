"""Definition of the DepthsMaterialsColumnPairs class."""

import math
from dataclasses import dataclass

import fitz
from stratigraphy.depthcolumn.depthcolumn import DepthColumn
from stratigraphy.lines.line import TextWord


@dataclass
class DepthsMaterialsColumnPair:
    """A class to represent pairs of depth columns and material descriptions."""

    depth_column: DepthColumn | None
    material_description_rect: fitz.Rect
    page: int

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

    def score_column_match(self, all_words: list[TextWord] | None = None) -> float:
        """Scores the match between a depth column and a material description.

        Args:
            all_words (list[TextWord] | None, optional): List of the available text words. Defaults to None.

        Returns:
            float: The score of the match.
        """
        rect = self.depth_column.rect()
        top = rect.y0
        bottom = rect.y1
        right = rect.x1
        distance = (
            abs(top - self.material_description_rect.y0)
            + abs(bottom - self.material_description_rect.y1)
            + abs(right - self.material_description_rect.x0)
        )

        height = bottom - top

        noise_count = self.depth_column.noise_count(all_words) if all_words else 0

        return (height - distance) * math.pow(0.8, noise_count)
