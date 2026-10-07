"""Layer class definition."""

from dataclasses import dataclass
from decimal import Decimal

import pymupdf

from extraction.features.stratigraphy.interval.interval import Interval
from extraction.utils.json import JsonSerializableRect
from swissgeol_doc_processing.text.textblock import MaterialDescription
from swissgeol_doc_processing.utils.data_extractor import ExtractedFeature
from swissgeol_doc_processing.utils.file_utils import parse_text


@dataclass
class LayerDepthsEntry:
    """Represents the upper or lower limit of a layer, used specifically for visualization and evaluation.

    Unlike `DepthColumnEntry` in `sidebarentry.py`, this class holds the extracted depth information,
    rather than being involved throughout the extraction process.
    """

    value: Decimal
    rect: JsonSerializableRect | None
    page_number: int

    def __repr__(self):
        return f"{self.value}"


class LayerDepths(ExtractedFeature):
    """Represents the start and end depth boundaries of a layer.

    Unlike the class `Interval` from `interval.py`, which is used in logical depth computations and extraction flow,
    the class `LayerDepths` is primarily used for data representation, visualiation and evaluation. It holds two
    extracted depth entries (`LayerDepthsEntry`), as opposed to Interval holding two `DepthColumnEntry`.
    """

    start: LayerDepthsEntry | None
    end: LayerDepthsEntry | None

    def get_line_anchor(self, page_number) -> pymupdf.Point | None:
        """Get the anchor point for the line connecting the start and end depths.

        Args:
            page_number (int): The page number for which to get the anchor point.

        Returns:
            pymupdf.Point | None: The anchor point for the line, or None if not applicable.
        """
        if self.start and self.start.rect and self.end:
            if self.start.page_number == self.end.page_number:
                return pymupdf.Point(
                    max(self.start.rect.x1, self.end.rect.x1), (self.start.rect.y0 + self.end.rect.y1) / 2
                )
            else:
                # Cross-page layers: no connector line (the borehole span marker already shows continuity)
                return None
        elif self.start and self.start.rect:
            return pymupdf.Point(self.start.rect.x1, self.start.rect.y1)
        elif self.end:
            return pymupdf.Point(self.end.rect.x1, self.end.rect.y0)

    def get_background_rect(self, page_number: int) -> pymupdf.Rect | None:
        """Get the background rectangle for the layer depths.

        Args:
            page_number (int): The page number for which to get the background rectangle.

        Returns:
            pymupdf.Rect | None: The background rectangle for the layer depths, or None if not applicable.
        """
        if not (self.start and self.start.rect and self.end):
            return None
        if self.start.page_number != self.end.page_number:
            return None
        elif self.start.rect.y1 < self.end.rect.y0:
            rect = pymupdf.Rect(
                self.start.rect.x0, self.start.rect.y1, max(self.start.rect.x1, self.end.rect.x1), self.end.rect.y0
            )
        else:
            return None

        return rect

    @classmethod
    def from_interval(cls, interval: Interval) -> "LayerDepths":
        """Converts an Interval to a LayerDepths object.

        Args:
            interval (Interval): an AAboveBInterval or AToBInterval.

        Returns:
            LayerDepths: the corresponding LayerDepths object.
        """
        start = interval.start
        end = interval.end
        return cls(
            start=LayerDepthsEntry(start.value, start.rect, start.page_number) if start else None,
            end=LayerDepthsEntry(end.value, end.rect, end.page_number) if end else None,
        )

    def is_valid_depth_interval(self, start: float, end: float) -> bool:
        """Validate if self and the depth interval start-end match.

        Args:
            start (float): The start value of the interval.
            end (float): The end value of the interval.

        Returns:
            bool: True if the depth intervals match, False otherwise.
        """
        if (self.start is not None) and (self.end is not None):
            return start == self.start.value and end == self.end.value

        return False


class Layer(ExtractedFeature):
    """Represents a finalized layer prediction in a borehole profile.

    A `Layer` combines a material description with its associated depth information,
    typically derived from a cleaned and validated `Sidebar` after extraction. It is the
    main data structure used for representing stratigraphy in the `ExtractedBorehole` class.
    """

    material_description: MaterialDescription
    depths: LayerDepths | None

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Layer(material_description={self.material_description}, depths={self.depths})"

    def description_nonempty(self) -> bool:
        return parse_text(self.material_description.text) != ""

    def depth_nonempty(self) -> bool:
        """Check whether both start and end depths are defined.

        Returns:
            bool: True if both `depths.start` and `depths.end` are defined, False otherwise.
        """
        return bool(self.depths and self.depths.start and self.depths.end)
