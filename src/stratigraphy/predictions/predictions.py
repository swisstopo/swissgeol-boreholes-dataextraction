"""Prediction classes for stratigraphy."""

import abc
from dataclasses import dataclass

import fitz
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInformationOnPage
from stratigraphy.layer.layer import LayerPrediction
from stratigraphy.metadata.metadata import (
    BoreholeMetadata,
    BoreholeMetadataMetrics,
    FileBoreholeMetadataMetrics,
    Metrics,
)
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.interval import BoundaryInterval
from stratigraphy.util.predictions import _create_textblock_object


@dataclass
class DepthMaterialColumnPair(metaclass=abc.ABCMeta):
    """Depth Material Column Pair class definition."""

    # TODO: Add depth material column pair properties


@dataclass
class FilePredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy from a single file."""

    groundwater: list[GroundwaterInformationOnPage] | None = None
    layers: list[LayerPrediction] = None
    depths_materials_column_pairs: list[DepthMaterialColumnPair] = None
    filename: str = None

    @staticmethod
    def extract_layers(json_layer_data: dict) -> list[LayerPrediction]:
        """Extracts layers from a JSON object.

        Args:
            json_data (dict): The JSON object.

        Returns:
            list[LayerPrediction]: The layers.
        """
        layers = []
        for layer in json_layer_data:
            material_prediction = _create_textblock_object(layer["material_description"]["lines"])
            if "depth_interval" in layer:
                start = (
                    DepthColumnEntry(
                        value=layer["depth_interval"]["start"]["value"],
                        rect=fitz.Rect(layer["depth_interval"]["start"]["rect"]),
                        page_number=layer["depth_interval"]["start"]["page"],
                    )
                    if layer["depth_interval"]["start"] is not None
                    else None
                )
                end = (
                    DepthColumnEntry(
                        value=layer["depth_interval"]["end"]["value"],
                        rect=fitz.Rect(layer["depth_interval"]["end"]["rect"]),
                        page_number=layer["depth_interval"]["end"]["page"],
                    )
                    if layer["depth_interval"]["end"] is not None
                    else None
                )

                depth_interval_prediction = BoundaryInterval(start=start, end=end)
                layer_predictions = LayerPrediction(
                    material_description=material_prediction, depth_interval=depth_interval_prediction
                )
            else:
                layer_predictions = LayerPrediction(material_description=material_prediction, depth_interval=None)

            layers.append(layer_predictions)

        return layers


@dataclass
class ExtractedFileInformation(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    single_file_predictions: FilePredictions = None
    metadata: BoreholeMetadata = None
    filename: str = None


class StratigraphyPredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    extracted_file_information: list[ExtractedFileInformation] = []

    def add_extracted_file_information(self, extracted_file_information: ExtractedFileInformation):
        """Adds extracted file information.

        Args:
            extracted_file_information (ExtractedFileInformation): The extracted file information.
        """
        self.extracted_file_information.append(extracted_file_information)

    def evaluate(self, ground_truth_path) -> BoreholeMetadataMetrics:
        """Evaluates the predictions against the ground truth."""
        metadata_metrics: list[FileBoreholeMetadataMetrics] = []

        # Iterate over the extracted file information for each file
        for extracted_file_information in self.extracted_file_information:
            # Compute the metadata metrics
            metadata_metrics.append(extracted_file_information.metadata.evaluate(ground_truth_path))

        # Compute the average metadata metrics
        tp_coords = sum([metric.coordinates_metrics.tp for metric in metadata_metrics])
        fp_coords = sum([metric.coordinates_metrics.fp for metric in metadata_metrics])
        fn_coords = sum([metric.coordinates_metrics.fn for metric in metadata_metrics])

        tp_elev = sum([metric.elevation_metrics.tp for metric in metadata_metrics])
        fp_elev = sum([metric.elevation_metrics.fp for metric in metadata_metrics])
        fn_elev = sum([metric.elevation_metrics.fn for metric in metadata_metrics])

        return BoreholeMetadataMetrics(
            coordinates_metrics=Metrics(tp=tp_coords, fp=fp_coords, fn=fn_coords, feature_name="coordinates"),
            elevation_metrics=Metrics(tp=tp_elev, fp=fp_elev, fn=fn_elev, feature_name="elevation"),
        )

    def to_json(self) -> dict:
        """Converts the object to a dictionary. Index is the filename.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            extracted_file_information.filename: {
                "metadata": extracted_file_information.metadata.to_json(),
                "layers": [layer.to_json() for layer in extracted_file_information.single_file_predictions.layers],
                "depths_materials_column_pairs": extracted_file_information.single_file_predictions.depths_materials_column_pairs,
                "groundwater": [
                    groundwater_on_page.to_json()
                    for groundwater_on_page in extracted_file_information.single_file_predictions.groundwater
                ],
            }
            for extracted_file_information in self.extracted_file_information
        }
