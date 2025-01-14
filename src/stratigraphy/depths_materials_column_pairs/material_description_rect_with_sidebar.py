"""Definition of the MaterialDescriptionRectWithSidebar class."""

import logging
import math
from dataclasses import dataclass

import fitz
from stratigraphy.lines.line import TextWord
from stratigraphy.sidebar import Sidebar

logger = logging.getLogger(__name__)


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
            float: The score of the match.
        """
        rect = self.sidebar.rect()
        top_sidebar = rect.y0
        bottom_sidebar = rect.y1
        right_sidebar = rect.x1

        vertical_distance = abs(top_sidebar - self.material_description_rect.y0) + abs(
            bottom_sidebar - self.material_description_rect.y1
        )
        horizontal_distance = abs(right_sidebar - self.material_description_rect.x0)
        distance = 1.5 * vertical_distance + horizontal_distance

        height_sidebar = bottom_sidebar - top_sidebar
        noise_count = self.sidebar.noise_count(all_words) if all_words else 0
        # noise_penalty = noise_count**3

        # return (height_sidebar - distance - noise_penalty)
        # TODO: check which scoring system will have max accuracy
        return (height_sidebar - distance) * math.pow(0.95, noise_count)
