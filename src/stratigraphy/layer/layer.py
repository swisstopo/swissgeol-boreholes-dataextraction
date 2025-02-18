"""Layer class definition."""

from dataclasses import dataclass

import fitz
from stratigraphy.data_extractor.data_extractor import ExtractedFeature, FeatureOnPage
from stratigraphy.depth import Interval
from stratigraphy.text.textblock import MaterialDescription, TextBlock
from stratigraphy.util.util import parse_text


@dataclass
class LayerDepthsEntry:
    """Class for the data about the upper or lower limit of a layer, as required for visualiation and evaluation."""

    value: float
    rect: fitz.Rect

    def to_json(self):
        """Convert the LayerDepthsEntry object to a JSON serializable format."""
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
        }

    @classmethod
    def from_json(cls, data: dict) -> "LayerDepthsEntry":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths entry.

        Returns:
            DepthColumnEntry: the corresponding LayerDepthsEntry object.
        """
        return cls(value=data["value"], rect=fitz.Rect(data["rect"]))


@dataclass
class LayerDepths:
    """Class for the data about the depths of a layer that are necessary for visualiation and evaluation."""

    start: LayerDepthsEntry | None
    end: LayerDepthsEntry | None

    @property
    def line_anchor(self) -> fitz.Point | None:
        if self.start and self.end:
            return fitz.Point(max(self.start.rect.x1, self.end.rect.x1), (self.start.rect.y0 + self.end.rect.y1) / 2)
        elif self.start:
            return fitz.Point(self.start.rect.x1, self.start.rect.y1)
        elif self.end:
            return fitz.Point(self.end.rect.x1, self.end.rect.y0)

    @property
    def background_rect(self) -> fitz.Rect | None:
        if self.start and self.end and self.start.rect.y1 < self.end.rect.y0:
            return fitz.Rect(
                self.start.rect.x0, self.start.rect.y1, max(self.start.rect.x1, self.end.rect.x1), self.end.rect.y0
            )

    def to_json(self):
        """Convert the LayerDepths object to a JSON serializable format."""
        return {"start": self.start.to_json() if self.start else None, "end": self.end.to_json() if self.end else None}

    @classmethod
    def from_json(cls, data: dict) -> "LayerDepths":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths.

        Returns:
            DepthColumnEntry: the corresponding LayerDepths object.
        """
        return cls(
            start=LayerDepthsEntry.from_json(data["start"]) if data["start"] else None,
            end=LayerDepthsEntry.from_json(data["end"]) if data["end"] else None,
        )

    @classmethod
    def from_interval(cls, interval: Interval) -> "LayerDepths":
        """Converts an Interval to a LayerDepths object.

        Args:
            interval (Interval): an AAboveBInterval or AToBInterval.

        Returns:
            LayerDepths: the corresponding LayerDepths object.
        """
        return cls(
            start=LayerDepthsEntry(interval.start.value, interval.start.rect) if interval.start else None,
            end=LayerDepthsEntry(interval.end.value, interval.end.rect) if interval.end else None,
        )


@dataclass
class Layer(ExtractedFeature):
    """A class to represent predictions for a single layer."""

    material_description: FeatureOnPage[MaterialDescription]
    depths: LayerDepths | None

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Layer(material_description={self.material_description}, depths={self.depths})"

    def description_nonempty(self) -> bool:
        return parse_text(self.material_description.feature.text) != ""

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "material_description": self.material_description.to_json() if self.material_description else None,
            "depths": self.depths.to_json() if self.depths else None,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Layer":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionarie representing the layer.

        Returns:
            list[LayerPrediction]: A list of LayerPrediction objects.
        """
        material_prediction = FeatureOnPage.from_json(data["material_description"], MaterialDescription)
        depths = LayerDepths.from_json(data["depths"]) if ("depths" in data and data["depths"] is not None) else None

        return Layer(material_description=material_prediction, depths=depths)


@dataclass
class LayersInDocument:
    """A class to represent predictions for a single document."""

    layers: list[Layer]
    filename: str


@dataclass
class IntervalBlockPair:
    """Represent the data for a single layer in the borehole profile.

    This consist of a material description (represented as a text block) and a depth interval (if available).
    """

    depth_interval: Interval | None
    block: TextBlock
