"""Definition of the MaterialDescriptionRectWithSidebar class."""

import math
from dataclasses import dataclass

import pymupdf
from stratigraphy.sidebar.sidebar import Sidebar


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
