"""This module contains classes for predictions."""

import logging
import os
from pathlib import Path

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetrics, OverallMetricsCatalog
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPairs
from stratigraphy.evaluation.evaluation_dataclasses import Metrics, OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.evaluation.utility import find_matching_layer
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument
from stratigraphy.layer.layer import Layer, LayersInDocument, LayersOnPage
from stratigraphy.metadata.metadata import BoreholeMetadata, OverallBoreholeMetadata
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers: LayersInDocument,
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater: GroundwaterInDocument,
        depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs],
    ):
        self.layers: LayersInDocument = layers
        self.depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs] = depths_materials_columns_pairs
        self.file_name: str = file_name
        self.metadata: BoreholeMetadata = metadata
        self.groundwater: GroundwaterInDocument = groundwater

    def convert_to_ground_truth(self):
        """Convert the predictions to ground truth format.

        This method is meant to be used in combination with the create_from_label_studio method.
        It converts the predictions to ground truth format, which can then be used for evaluation.

        NOTE: This method should be tested before using it to create new ground truth.

        Returns:
            dict: The predictions in ground truth format.
        """
        ground_truth = {self.file_name: {"metadata": self.metadata}}
        layers = []
        for layer in self.layers.get_all_layers():
            material_description = layer.material_description.text
            depth_interval = {
                "start": layer.depth_interval.start.value if layer.depth_interval.start else None,
                "end": layer.depth_interval.end.value if layer.depth_interval.end else None,
            }
            layers.append({"material_description": material_description, "depth_interval": depth_interval})
        ground_truth[self.file_name]["layers"] = layers
        if self.metadata is not None and self.metadata.coordinates is not None:
            ground_truth[self.file_name]["metadata"] = {
                "coordinates": {
                    "E": self.metadata.coordinates.east.coordinate_value,
                    "N": self.metadata.coordinates.north.coordinate_value,
                }
            }
        return ground_truth

    def evaluate(self, ground_truth: dict):
        """Evaluate the predictions against the ground truth.

        Args:
            ground_truth (dict): The ground truth for the file.
        """
        # TODO: Call the evaluator for Layers instead
        self.evaluate_layers(ground_truth["layers"])

    def evaluate_layers(self, ground_truth_layers: list):
        """Evaluate all layers of the predictions against the ground truth.

        Args:
            ground_truth_layers (list): The ground truth layers for the file.
        """
        unmatched_layers = ground_truth_layers.copy()
        for layer in self.layers.get_all_layers():
            match, depth_interval_is_correct = find_matching_layer(layer, unmatched_layers)
            if match:
                layer.material_is_correct = True
                layer.depth_interval_is_correct = depth_interval_is_correct
            else:
                layer.material_is_correct = False
                layer.depth_interval_is_correct = None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "metadata": self.metadata.to_json(),
            "layers": [layer.to_json() for layer in self.layers.get_all_layers()] if self.layers is not None else [],
            "depths_materials_column_pairs": [dmc_pair.to_json() for dmc_pair in self.depths_materials_columns_pairs]
            if self.depths_materials_columns_pairs is not None
            else [],
            "page_dimensions": self.metadata.page_dimensions,  # TODO: Remove, already in metadata
            "groundwater": self.groundwater.to_json() if self.groundwater is not None else [],
            "file_name": self.file_name,
        }


class OverallFilePredictions:
    """A class to represent predictions for all files."""

    def __init__(self) -> None:
        """Initializes the OverallFilePredictions object."""
        self.file_predictions_list: list[FilePredictions] = []

    def add_file_predictions(self, file_predictions: FilePredictions) -> None:
        """Add file predictions to the list of file predictions.

        Args:
            file_predictions (FilePredictions): The file predictions to add.
        """
        self.file_predictions_list.append(file_predictions)

    def get_metadata_as_dict(self) -> None:
        """Returns the metadata of the predictions as a dictionary."""
        return {
            file_prediction.file_name: file_prediction.metadata.to_json()
            for file_prediction in self.file_predictions_list
        }

    def to_json(self) -> dict:
        """Converts the object to a dictionary by merging individual file predictions.

        Returns:
            dict: A dictionary representation of the object.
        """
        return {fp.file_name: fp.to_json() for fp in self.file_predictions_list}

    @classmethod
    def from_json(cls, prediction_from_file: dict) -> "OverallFilePredictions":
        """Converts a dictionary to an object.

        Args:
            prediction_from_file (dict): A dictionary representing the predictions.

        Returns:
            OverallFilePredictions: The object.
        """
        overall_file_predictions = OverallFilePredictions()
        for file_name, file_data in prediction_from_file.items():
            metadata = BoreholeMetadata.from_json(file_data["metadata"], file_name)

            layers = Layer.from_json(file_data["layers"])
            layers_on_page = LayersOnPage(layers_on_page=layers)
            layers_in_doc = LayersInDocument(
                layers_in_document=[layers_on_page], filename=file_name
            )  # TODO: This is a bit of a hack as we do not seem to save the page of the layer

            depths_materials_columns_pairs = [
                DepthsMaterialsColumnPairs.from_json(dmc_pair)
                for dmc_pair in file_data["depths_materials_column_pairs"]
            ]

            groundwater_entries = [FeatureOnPage.from_json(entry, Groundwater) for entry in file_data["groundwater"]]
            groundwater_in_document = GroundwaterInDocument(groundwater=groundwater_entries, filename=file_name)
            overall_file_predictions.add_file_predictions(
                FilePredictions(
                    layers=layers_in_doc,
                    file_name=file_name,
                    metadata=metadata,
                    depths_materials_columns_pairs=depths_materials_columns_pairs,
                    groundwater=groundwater_in_document,
                )
            )
        return overall_file_predictions

    ############################################################################################################
    ### Evaluation methods
    ############################################################################################################

    def evaluate_metadata_extraction(self, ground_truth_path: Path) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            ground_truth_path (Path): The path to the ground truth file.
        """
        metadata_per_file: OverallBoreholeMetadata = OverallBoreholeMetadata()

        for file_prediction in self.file_predictions_list:
            metadata_per_file.add_metadata(file_prediction.metadata)
        return MetadataEvaluator(metadata_per_file, ground_truth_path).evaluate()

    def evaluate_borehole_extraction(self, ground_truth_path: Path) -> OverallMetricsCatalog | None:
        """Evaluate the borehole extraction predictions.

        Args:
            ground_truth_path (Path): The path to the ground truth file.

        Returns:
            OverallMetricsCatalogue: A OverallMetricsCatalog that maps a metrics name to the corresponding
            OverallMetrics object. If no ground truth is available, None is returned.
        """
        ############################################################################################################
        ### Load the ground truth data for the borehole extraction
        ############################################################################################################
        ground_truth = None
        if ground_truth_path and os.path.exists(ground_truth_path):  # for inference no ground truth is available
            ground_truth = GroundTruth(ground_truth_path)
        else:
            logger.warning("Ground truth file not found.")

        ############################################################################################################
        ### Evaluate the borehole extraction
        ############################################################################################################
        number_of_truth_values = {}
        for file_predictions in self.file_predictions_list:
            if ground_truth:
                ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
                if ground_truth_for_file:
                    file_predictions.evaluate(ground_truth_for_file)
                    number_of_truth_values[file_predictions.file_name] = len(ground_truth_for_file["layers"])

        if number_of_truth_values:
            all_metrics = self.evaluate_layer_extraction(number_of_truth_values)

            groundwater_entries = [file_prediction.groundwater for file_prediction in self.file_predictions_list]
            overall_groundwater_metrics = GroundwaterEvaluator(groundwater_entries, ground_truth_path).evaluate()
            all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
            all_metrics.groundwater_depth_metrics = (
                overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
            )
            return all_metrics
        else:
            logger.warning("Ground truth file not found. Skipping evaluation.")
            return None

    def evaluate_layer_extraction(self, number_of_truth_values: dict) -> OverallMetricsCatalog:
        """Calculate F1, precision and recall for the predictions.

        Calculate F1, precision and recall for the individual documents as well as overall.
        The individual document metrics are returned as a DataFrame.

        Args:
            number_of_truth_values (dict): The number of layer ground truth values per file.

        Returns:
            OverallMetricsCatalog: A dictionary that maps a metrics name to the corresponding OverallMetrics object
        """
        # create predictions by language
        languages = set(fp.metadata.language for fp in self.file_predictions_list)
        predictions_by_language = {language: OverallFilePredictions() for language in languages}

        all_metrics = OverallMetricsCatalog(languages=languages)
        all_metrics.layer_metrics = get_layer_metrics(self, number_of_truth_values)
        all_metrics.depth_interval_metrics = get_depth_interval_metrics(self)

        for file_predictions in self.file_predictions_list:
            language = file_predictions.metadata.language
            if language in predictions_by_language:
                predictions_by_language[language].add_file_predictions(file_predictions)

        for language, language_predictions in predictions_by_language.items():
            language_number_of_truth_values = {
                prediction.file_name: number_of_truth_values[prediction.file_name]
                for prediction in language_predictions.file_predictions_list
            }

            setattr(
                all_metrics,
                f"{language}_layer_metrics",
                get_layer_metrics(language_predictions, language_number_of_truth_values),
            )
            setattr(
                all_metrics, f"{language}_depth_interval_metrics", get_depth_interval_metrics(language_predictions)
            )

        logger.info("Macro avg:")
        logger.info(
            "F1: %.1f%%, precision: %.1f%%, recall: %.1f%%, depth_interval_accuracy: %.1f%%",
            all_metrics.layer_metrics.macro_f1() * 100,
            all_metrics.layer_metrics.macro_precision() * 100,
            all_metrics.layer_metrics.macro_recall() * 100,
            all_metrics.depth_interval_metrics.macro_precision() * 100,
        )

        return all_metrics


def get_layer_metrics(predictions: OverallFilePredictions, number_of_truth_values: dict) -> OverallMetrics:
    """Calculate F1, precision and recall for the layer predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    # TODO: Try to move this to the LayerPrediction class

    Args:
        predictions (OverallFilePredictions): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        OverallMetrics: the metrics for the layers
    """
    layer_metrics = OverallMetrics()

    for file_prediction in predictions.file_predictions_list:
        hits = 0
        for layer in file_prediction.layers.get_all_layers():
            if layer.material_is_correct:
                hits += 1
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")
        layer_metrics.metrics[file_prediction.file_name] = Metrics(
            tp=hits,
            fp=len(file_prediction.layers.get_all_layers()) - hits,
            fn=number_of_truth_values[file_prediction.file_name] - hits,
        )

    return layer_metrics


def get_depth_interval_metrics(predictions: OverallFilePredictions) -> OverallMetrics:
    """Calculate F1, precision and recall for the depth interval predictions.

    # TODO: Try to move this to the LayerPrediction class

    Calculate F1, precision and recall for the individual documents as well as overall.

    Depth interval accuracy is not calculated for layers with incorrect material predictions.

    Args:
        predictions (OverallFilePredictions): The predictions.

    Returns:
        OverallMetrics: the metrics for the depth intervals
    """
    depth_interval_metrics = OverallMetrics()

    for file_prediction in predictions.file_predictions_list:
        depth_interval_hits = 0
        depth_interval_occurrences = 0
        for layer in file_prediction.layers.get_all_layers():
            if layer.material_is_correct:
                if layer.depth_interval_is_correct is not None:
                    depth_interval_occurrences += 1
                if layer.depth_interval_is_correct:
                    depth_interval_hits += 1

        if depth_interval_occurrences > 0:
            depth_interval_metrics.metrics[file_prediction.file_name] = Metrics(
                tp=depth_interval_hits, fp=depth_interval_occurrences - depth_interval_hits, fn=0
            )

    return depth_interval_metrics
