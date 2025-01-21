"""Definition of the MaterialDescriptionRectWithSidebar class."""

import math
from dataclasses import dataclass

import fitz
from stratigraphy.sidebar import Sidebar


@dataclass
class MaterialDescriptionRectWithSidebar:
    """A class to represent pairs of sidebar and material description rectangle."""

    sidebar: Sidebar | None
    material_description_rect: fitz.Rect
    noise_count: int = 0

    @property
    def score_match(self) -> float:
        """Scores the match between a sidebar and a material description.

        Returns:
            float: The score of the match.
        """
        rect = self.sidebar.rect()
        top = rect.y0
        bottom = rect.y1
        right = rect.x1
        distance = (
            abs(top - self.material_description_rect.y0)
            + abs(bottom - self.material_description_rect.y1)
            + abs(right - self.material_description_rect.x0)
        )

        height = bottom - top

        return (height - distance) * math.pow(0.8, self.noise_count)
