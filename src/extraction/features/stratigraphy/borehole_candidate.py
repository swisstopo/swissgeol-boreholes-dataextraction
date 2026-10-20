"""Module containing a dataclass for scored candidate boreholes."""

import math
from dataclasses import dataclass

import pymupdf

from extraction.features.extracted_borehole import ExtractedBorehole
from extraction.features.stratigraphy.sidebar.classes.layer_identifier_sidebar import LayerIdentifierSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar, SidebarNoise


@dataclass
class BoreholeCandidate:
    """Dataclass for a potential borehole created from a material description rect and optional sidebar."""

    borehole: ExtractedBorehole
    material_description_rect: pymupdf.Rect
    sidebar: Sidebar | None
    score: float

    @property
    def bounding_box(self) -> pymupdf.Rect:
        """Creates a bounding box around the candidate.

        The box is the union of the material description rect and the sidebar rect (if present).
        """
        bbox = self.material_description_rect
        if self.sidebar:
            bbox = bbox | self.sidebar.rect
        return bbox

    @classmethod
    def from_pair(cls, borehole: ExtractedBorehole, sidebar_noise: SidebarNoise | None) -> "BoreholeCandidate":
        sidebar = sidebar_noise.sidebar if sidebar_noise else None

        # We boost the indicator sidebar, because they usually filter out some description lines, which should not have
        # a negative impact on the matching score. Therefore, we count an additional description line for each
        # indicator.
        indicator_count = len(sidebar.entries) if sidebar and isinstance(sidebar, LayerIdentifierSidebar) else 0

        layers_without_description = max(
            0, sum(1 for layer in borehole.predictions if not layer.description_nonempty()) - indicator_count
        )
        layers_without_description_penalty = 1 / (1 + layers_without_description)

        description_lines_score = math.log(
            sum(len(layer.material_description.lines) for layer in borehole.predictions) + indicator_count
        )

        rect = pymupdf.Rect()
        for layer in borehole.predictions:
            for line in layer.material_description.lines:
                rect.include_rect(line.rect)

        # boreholes without a sidebar always score 0.0
        score_match = sidebar_noise.score_match(rect) if sidebar_noise else 0.0
        score = score_match * layers_without_description_penalty * description_lines_score

        return cls(borehole, rect, sidebar, score)
