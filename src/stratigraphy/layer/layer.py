"""Layer class definition."""

from dataclasses import dataclass

import fitz
from stratigraphy.data_extractor.data_extractor import ExtractedFeature, FeatureOnPage
from stratigraphy.depth import AAboveBInterval, Interval
from stratigraphy.sidebar.sidebarentry import DepthColumnEntry
from stratigraphy.text.textblock import MaterialDescription, TextBlock
from stratigraphy.util.util import parse_text


@dataclass
class Layer(ExtractedFeature):
    """A class to represent predictions for a single layer."""

    material_description: FeatureOnPage[MaterialDescription]
    depth_interval: AAboveBInterval | None

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Layer(material_description={self.material_description}, depth_interval={self.depth_interval})"

    def description_nonempty(self) -> bool:
        return parse_text(self.material_description.feature.text) != ""

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "material_description": self.material_description.to_json() if self.material_description else None,
            "depth_interval": self.depth_interval.to_json() if self.depth_interval else None,
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
        if "depth_interval" in data and data["depth_interval"] is not None:
            depth_interval = data.get("depth_interval", {})
            start_data = depth_interval.get("start")
            end_data = depth_interval.get("end")
            start = (
                DepthColumnEntry(value=start_data["value"], rect=fitz.Rect(start_data["rect"]))
                if start_data is not None
                else None
            )
            end = (
                DepthColumnEntry(value=end_data["value"], rect=fitz.Rect(end_data["rect"]))
                if end_data is not None
                else None
            )

            depth_interval_prediction = AAboveBInterval(start=start, end=end)
        else:
            depth_interval_prediction = None

        return Layer(material_description=material_prediction, depth_interval=depth_interval_prediction)


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
