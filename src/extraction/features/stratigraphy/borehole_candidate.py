"""Module containing a dataclass for scored candidate boreholes."""

from dataclasses import dataclass

import pymupdf

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole
from extraction.features.stratigraphy.layer.page_bounding_boxes import MaterialDescriptionRectWithSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar


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
    def from_pair(cls, borehole: ExtractedBorehole, pair: MaterialDescriptionRectWithSidebar) -> "BoreholeCandidate":
        layers_without_description = sum(1 for layer in borehole.predictions if not layer.description_nonempty())
        layers_without_description_penalty = 1 / (1 + layers_without_description)

        description_lines_score = sum(len(layer.material_description.lines) for layer in borehole.predictions)

        score = pair.score_match * layers_without_description_penalty * description_lines_score
        return cls(borehole, pair.material_description_rect, pair.sidebar, score)
