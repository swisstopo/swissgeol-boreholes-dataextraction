"""Definition of the MaterialDescriptionRectWithSidebar class."""

import math
from dataclasses import dataclass

import fitz
from stratigraphy.lines.line import TextWord
from stratigraphy.sidebar import Sidebar


@dataclass
class MaterialDescriptionRectWithSidebar:
    """A class to represent pairs of sidebar and material description rectangle."""

    sidebar: Sidebar | None
    material_description_rect: fitz.Rect

    def score_match(self, all_words: list[TextWord] | None = None) -> float:
        """Scores the match between a sidebar and a material description.

        Args:
            all_words (list[TextWord] | None, optional): List of the available text words. Defaults to None.

        Returns:
            float: The score of the match. Better matches have a higher score value.
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

        noise_count = self.sidebar.noise_count(all_words) if all_words else 0
        return (height - distance) * math.pow(0.8, 10 * noise_count / len(self.sidebar.entries))
