"""Layer class definition."""

from dataclasses import dataclass

from borehole_extraction.extraction.stratigraphy.depth.interval import DepthInterval
from borehole_extraction.extraction.stratigraphy.depths_materials_pairs.bounding_boxes import PageBoundingBoxes
from borehole_extraction.extraction.util_extraction.data_extractor.data_extractor import (
    ExtractedFeature,
    FeatureOnPage,
)
from borehole_extraction.extraction.util_extraction.text.textblock import MaterialDescription
from general_utils.file_utils import parse_text


@dataclass
class Layer(ExtractedFeature):
    """A class to represent predictions for a single layer."""

    material_description: FeatureOnPage[MaterialDescription]
    depths: DepthInterval | None

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
        depths = DepthInterval.from_json(data["depths"]) if ("depths" in data and data["depths"] is not None) else None

        return Layer(material_description=material_prediction, depths=depths)


@dataclass
class LayersInBorehole:
    """Represent the data for all layers in a borehole profile."""

    layers: list[Layer]

    def to_json(self):
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return [layer.to_json() for layer in self.layers]

    @classmethod
    def from_json(cls, json_object) -> "LayersInBorehole":
        """Extract a LayersInBorehole object from a json dictionary.

        Args:
            json_object (dict): The object as a dictionary.

        Returns:
            LayersInBorehole: The LayersInBorehole object.
        """
        return cls([Layer.from_json(layer_data) for layer_data in json_object])


@dataclass
class ExtractedBorehole:
    """A class to store the extracted information of one single borehole."""

    predictions: list[Layer]
    bounding_boxes: list[PageBoundingBoxes]  # one for each page that the borehole spans


class LayersInDocument:
    """A class to represent predictions for a single document.

    It contains a list of LayersInBorehole, not just a list of Layer.
    """

    def __init__(self, boreholes_layers: list[ExtractedBorehole], filename: str):
        self.boreholes_layers_with_bb = boreholes_layers
        self.filename = filename

    def assign_layers_to_boreholes(self, layer_predictions: list[ExtractedBorehole]):
        """SIMPLIFICATION: currently assumes that if there is more than one page, there is a single borehole.

        This means that only pdfs with one page could contain more than one borehole.
        This is the case for all example from zurich and geoquat,validation.
        If we want to be more robust, a matching should be done to determine which layers goes with which borehole.

        Args:
            layer_predictions (list[ExtractedBorehole]): List containing the a list of all layers of all boreholes
        """
        if not layer_predictions:
            return
        if not self.boreholes_layers_with_bb:
            # first page
            self.boreholes_layers_with_bb = layer_predictions
        else:
            # second page, use assumption, also fot the bounding boxes
            self.boreholes_layers_with_bb[0].bounding_boxes.extend(layer_predictions[0].bounding_boxes)
            self.boreholes_layers_with_bb[0].predictions.extend(layer_predictions[0].predictions)
