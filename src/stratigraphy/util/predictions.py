"""This module contains classes for predictions."""

import logging
import os
from pathlib import Path

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetrics, OverallMetricsCatalog
from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPairs
from stratigraphy.evaluation.evaluation_dataclasses import Metrics, OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.evaluation.utility import find_matching_layer
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInDocument
from stratigraphy.layer.layer import LayerPrediction
from stratigraphy.metadata.metadata import BoreholeMetadata, BoreholeMetadataList
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers: list[LayerPrediction],
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater: GroundwaterInDocument,
        depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs],
    ):
        self.layers: list[LayerPrediction] = layers
        self.depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs] = depths_materials_columns_pairs
        self.file_name = file_name
        self.metadata = metadata
        self.groundwater = groundwater

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
        for layer in self.layers:
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
        for layer in self.layers:
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
            self.file_name: {
                "metadata": self.metadata.to_json(),
                "layers": [layer.to_json() for layer in self.layers],
                "depths_materials_column_pairs": [
                    depths_materials_columns_pairs.to_json()
                    for depths_materials_columns_pairs in self.depths_materials_columns_pairs
                ],
                "page_dimensions": self.metadata.page_dimensions,
                # TODO: This should be removed. As already in metadata.
                "groundwater": [entry.to_json() for entry in self.groundwater.groundwater],
                "file_name": self.file_name,
            }
        }


class OverallFilePredictions:
    """A class to represent predictions for all files."""

    file_predictions_list: list[FilePredictions] = None

    def __init__(self):
        """Initializes the OverallFilePredictions object."""
        self.file_predictions_list = []

    def add_file_predictions(self, file_predictions: FilePredictions):
        """Add file predictions to the list of file predictions.

        Args:
            file_predictions (FilePredictions): The file predictions to add.
        """
        self.file_predictions_list.append(file_predictions)

    def export_metadata_to_json(self):
        """Export the metadata of the predictions to a json file.

        Args:
            output_file (str): The path to the output file.
        """
        return {
            file_prediction.file_name: file_prediction.metadata.to_json()
            for file_prediction in self.file_predictions_list
        }

    def to_json(self):
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {file_prediction.file_name: file_prediction.to_json() for file_prediction in self.file_predictions_list}

    def get_groundwater_entries(self) -> list[GroundwaterInDocument]:
        """Get the groundwater extractions from the predictions.

        Returns:
            List[GroundwaterInDocument]: The groundwater extractions.
        """
        return [file_prediction.groundwater for file_prediction in self.file_predictions_list]

    ############################################################################################################
    ### Evaluation methods
    ############################################################################################################

    def evaluate_metadata_extraction(self, ground_truth_path: Path) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        # TODO: Move to evaluator class

        Args:
            ground_truth_path (Path): The path to the ground truth file.
        """
        metadata_per_file = BoreholeMetadataList()

        for file_prediction in self.file_predictions_list:
            metadata_per_file.add_metadata(file_prediction.metadata)
        return MetadataEvaluator(metadata_per_file, ground_truth_path).evaluate()

    def evaluate_borehole_extraction(self, ground_truth_path: str) -> OverallMetricsCatalog:
        """Evaluate the borehole extraction predictions.

        Args:
            ground_truth_path (str): The path to the ground truth file.

        Returns:
            OverallMetricsCatalogue: A OverallMetricsCatalog that maps a metrics name to the corresponding
            OverallMetrics object
        """
        ############################################################################################################
        ### Load the ground truth data for the borehole extraction
        ############################################################################################################
        ground_truth = None
        if ground_truth_path and os.path.exists(ground_truth_path):  # for inference no ground truth is available
            ground_truth = GroundTruth(ground_truth_path)
        else:
            logging.warning("Ground truth file not found.")

        ############################################################################################################
        ### Evaluate the borehole extraction
        ############################################################################################################
        number_of_truth_values = {}
        for file_predictions in self.file_predictions_list:
            # prediction_object = FilePredictions.create_from_json(file_predictions, file_predictions.file_name)

            # predictions_objects[file_name] = prediction_object
            if ground_truth:
                ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
                if ground_truth_for_file:
                    file_predictions.evaluate(ground_truth_for_file)
                    number_of_truth_values[file_predictions.file_name] = len(ground_truth_for_file["layers"])

        if number_of_truth_values:
            all_metrics = self.evaluate_layer_extraction(number_of_truth_values)

            overall_groundwater_metrics = GroundwaterEvaluator(
                self.get_groundwater_entries(), ground_truth_path
            ).evaluate()
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
            predictions (dict): The FilePredictions objects.
            number_of_truth_values (dict): The number of layer ground truth values per file.

        Returns:
            OverallMetricsCatalog: A dictionary that maps a metrics name to the corresponding OverallMetrics object
        """
        all_metrics = OverallMetricsCatalog()
        all_metrics.set_layer_metrics(get_layer_metrics(self, number_of_truth_values))
        all_metrics.set_depth_interval_metrics(get_depth_interval_metrics(self))

        # create predictions by language
        predictions_by_language = {
            "de": OverallFilePredictions(),
            "fr": OverallFilePredictions(),
        }  # TODO: make this dynamic and why is this hardcoded?
        for file_predictions in self.file_predictions_list:
            language = file_predictions.metadata.language
            if language in predictions_by_language:
                predictions_by_language[language].add_file_predictions(file_predictions)

        for language, language_predictions in predictions_by_language.items():
            language_number_of_truth_values = {
                prediction.file_name: number_of_truth_values[prediction.file_name]
                for prediction in language_predictions.file_predictions_list
            }
            if language == "de":
                all_metrics.set_de_layer_metrics(
                    get_layer_metrics(language_predictions, language_number_of_truth_values)
                )
                all_metrics.set_de_depth_interval_metrics(get_depth_interval_metrics(language_predictions))
            elif language == "fr":
                all_metrics.set_fr_layer_metrics(
                    get_layer_metrics(language_predictions, language_number_of_truth_values)
                )
                all_metrics.set_fr_depth_interval_metrics(get_depth_interval_metrics(language_predictions))

        logging.info("Macro avg:")
        logging.info(
            "F1: %.1f%%, precision: %.1f%%, recall: %.1f%%, depth_interval_accuracy: %.1f%%",
            all_metrics.layer_metrics.macro_f1() * 100,
            all_metrics.layer_metrics.macro_precision() * 100,
            all_metrics.layer_metrics.macro_recall() * 100,
            all_metrics.depth_interval_metrics.macro_precision() * 100,
        )

        return all_metrics

    def get_metrics(self, field_key: str, field_name: str) -> OverallMetrics:
        """Get the metrics for a specific field in the predictions.

        Args:
            predictions (dict): The FilePredictions objects.
            field_key (str): The key to access the specific field in the prediction objects.
            field_name (str): The name of the field being evaluated.

        Returns:
            OverallMetrics: The requested OverallMetrics object.
        """
        overall_metrics = OverallMetrics()

        for file_prediction in self.file_predictions_list:
            overall_metrics.metrics[file_prediction.file_name] = getattr(file_prediction, field_key)[field_name]

        return overall_metrics


def get_layer_metrics(predictions: OverallFilePredictions, number_of_truth_values: dict) -> OverallMetrics:
    """Calculate F1, precision and recall for the layer predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    # TODO: Try to mode this to the LayerPrediction class

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        OverallMetrics: the metrics for the layers
    """
    layer_metrics = OverallMetrics()

    for file_prediction in predictions.file_predictions_list:
        hits = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                hits += 1
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")
        layer_metrics.metrics[file_prediction.file_name] = Metrics(
            tp=hits, fp=len(file_prediction.layers) - hits, fn=number_of_truth_values[file_prediction.file_name] - hits
        )

    return layer_metrics


def get_depth_interval_metrics(predictions: OverallFilePredictions) -> OverallMetrics:
    """Calculate F1, precision and recall for the depth interval predictions.

    # TODO: Try to mode this to the LayerPrediction class

    Calculate F1, precision and recall for the individual documents as well as overall.

    Depth interval accuracy is not calculated for layers with incorrect material predictions.

    Args:
        predictions (dict): The predictions.

    Returns:
        OverallMetrics: the metrics for the depth intervals
    """
    depth_interval_metrics = OverallMetrics()

    for file_prediction in predictions.file_predictions_list:
        depth_interval_hits = 0
        depth_interval_occurrences = 0
        for layer in file_prediction.layers:
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
