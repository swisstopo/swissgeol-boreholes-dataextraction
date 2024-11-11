"""This module contains classes for predictions."""

import logging

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetricsCatalog
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPair
from stratigraphy.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument
from stratigraphy.layer.layer import Layer, LayersInDocument
from stratigraphy.metadata.metadata import BoreholeMetadata, OverallBoreholeMetadata

logger = logging.getLogger(__name__)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers_in_document: LayersInDocument,
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater: GroundwaterInDocument,
        depths_materials_columns_pairs: list[DepthsMaterialsColumnPair],
    ):
        self.layers_in_document: LayersInDocument = layers_in_document
        self.depths_materials_columns_pairs: list[DepthsMaterialsColumnPair] = depths_materials_columns_pairs
        self.file_name: str = file_name
        self.metadata: BoreholeMetadata = metadata
        self.groundwater: GroundwaterInDocument = groundwater

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "metadata": self.metadata.to_json(),
            "layers": [layer.to_json() for layer in self.layers_in_document.layers],
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

    def get_metadata_as_dict(self) -> dict:
        """Returns the metadata of the predictions as a dictionary.

        Returns:
            dict: The metadata of the predictions as a dictionary.
        """
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

            layers = [Layer.from_json(data) for data in file_data["layers"]]
            layers_in_doc = LayersInDocument(layers=layers, filename=file_name)

            depths_materials_columns_pairs = [
                DepthsMaterialsColumnPair.from_json(dmc_pair)
                for dmc_pair in file_data["depths_materials_column_pairs"]
            ]

            groundwater_entries = [FeatureOnPage.from_json(entry, Groundwater) for entry in file_data["groundwater"]]
            groundwater_in_document = GroundwaterInDocument(groundwater=groundwater_entries, filename=file_name)
            overall_file_predictions.add_file_predictions(
                FilePredictions(
                    layers_in_document=layers_in_doc,
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

    def evaluate_metadata_extraction(self, ground_truth: GroundTruth) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            ground_truth (GroundTruth): The ground truth.
        """
        metadata_per_file: OverallBoreholeMetadata = OverallBoreholeMetadata()

        for file_prediction in self.file_predictions_list:
            metadata_per_file.add_metadata(file_prediction.metadata)
        return MetadataEvaluator(metadata_per_file, ground_truth).evaluate()

    def evaluate_geology(self, ground_truth: GroundTruth) -> OverallMetricsCatalog | None:
        """Evaluate the borehole extraction predictions.

        Args:
            ground_truth (GroundTruth): The ground truth.

        Returns:
            OverallMetricsCatalog: A OverallMetricsCatalog that maps a metrics name to the corresponding
            OverallMetrics object. If no ground truth is available, None is returned.
        """
        for file_predictions in self.file_predictions_list:
            ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
            if ground_truth_for_file:
                LayerEvaluator.evaluate_borehole(
                    file_predictions.layers_in_document.layers, ground_truth_for_file["layers"]
                )

        languages = set(fp.metadata.language for fp in self.file_predictions_list)
        all_metrics = OverallMetricsCatalog(languages=languages)

        evaluator = LayerEvaluator(
            [prediction.layers_in_document for prediction in self.file_predictions_list], ground_truth
        )
        all_metrics.layer_metrics = evaluator.get_layer_metrics()
        all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()

        layers_in_doc_by_language = {language: [] for language in languages}
        for file_prediction in self.file_predictions_list:
            layers_in_doc_by_language[file_prediction.metadata.language].append(file_prediction.layers_in_document)

        for language, layers_in_doc_list in layers_in_doc_by_language.items():
            evaluator = LayerEvaluator(layers_in_doc_list, ground_truth)
            setattr(
                all_metrics,
                f"{language}_layer_metrics",
                evaluator.get_layer_metrics(),
            )
            setattr(all_metrics, f"{language}_depth_interval_metrics", evaluator.get_depth_interval_metrics())

        logger.info("Macro avg:")
        logger.info(
            "F1: %.1f%%, precision: %.1f%%, recall: %.1f%%, depth_interval_precision: %.1f%%",
            all_metrics.layer_metrics.macro_f1() * 100,
            all_metrics.layer_metrics.macro_precision() * 100,
            all_metrics.layer_metrics.macro_recall() * 100,
            all_metrics.depth_interval_metrics.macro_precision() * 100,
        )

        groundwater_entries = [file_prediction.groundwater for file_prediction in self.file_predictions_list]
        overall_groundwater_metrics = GroundwaterEvaluator(groundwater_entries, ground_truth).evaluate()
        all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        all_metrics.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )
        return all_metrics
