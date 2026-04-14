"""Evaluator for borehole predictions against ground truth data."""

import logging

from core.benchmark_utils import Metrics
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.benchmark.metrics import OverallMetrics, OverallMetricsCatalog
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics, OverallBoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterEvaluator, OverallGroundwaterMetrics
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.evaluation.metadata_evaluator import MetadataEvaluator
from extraction.features.predictions.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    BoreholeLayersWithGroundTruth,
    BoreholeMetadataWithGroundTruth,
    FileGroundwaterWithGroundTruth,
    FileLayersWithGroundTruth,
    FileMetadataWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.predictions.predictions import AllBoreholePredictionsWithGroundTruth

logger = logging.getLogger(__name__)


class Evaluator:
    """Static utility class for matching borehole predictions with ground truth and computing evaluation metrics."""

    @staticmethod
    def match_with_ground_truth(
        file_predictions: FilePredictions, ground_truth: GroundTruth
    ) -> FilePredictionsWithGroundTruth:
        """Match the extracted boreholes with corresponding boreholes in the ground truth data.

        This is done by comparing the layers of the extracted boreholes with those in the groundtruth.

        Args:
            file_predictions (FilePredictions): File predictions.
            ground_truth (GroundTruth): The ground truth.

        Returns:
            AllBoreholePredictionsWithGroundTruth: all predictions per borehole with associated ground truth data.
        """
        boreholes = []
        ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
        if ground_truth_for_file:
            boreholes = LayerEvaluator.match_boreholes_to_ground_truth(file_predictions, ground_truth_for_file)
        return FilePredictionsWithGroundTruth(
            file_predictions.file_name, file_predictions.file_metadata.language, boreholes
        )

    @staticmethod
    def evaluate(file_predictions: FilePredictionsWithGroundTruth) -> tuple[BoreholeMetadataMetrics]:
        Evaluator.evaluate_layers(file_predictions)
        Evaluator.evaluate_metadata(file_predictions)
        Evaluator.evaluate_gw(file_predictions)

    @staticmethod
    def evaluate_layers(file_predictions: FilePredictionsWithGroundTruth) -> None:
        return LayerEvaluator.evaluate(
            file_predictions=FileLayersWithGroundTruth(
                file_predictions.filename,
                file_predictions.language,
                [
                    BoreholeLayersWithGroundTruth(
                        predictions.predictions.layers_in_borehole if predictions.predictions else None,
                        predictions.ground_truth.get("layers", []),
                    )
                    for predictions in file_predictions.boreholes
                ],
            )
        )

    @staticmethod
    def evaluate_metadata(
        file_predictions: FilePredictionsWithGroundTruth,
    ) -> BoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            file_predictions (FilePredictionsWithGroundTruth): Per-file borehole predictions
                paired with their ground truth data.

        Returns:
            BoreholeMetadataMetrics: the computed metrics for the metadata.
        """
        return MetadataEvaluator.evaluate(
            file_predictions=FileMetadataWithGroundTruth(
                file_predictions.filename,
                [
                    BoreholeMetadataWithGroundTruth(
                        predictions.predictions.metadata if predictions.predictions else None,
                        predictions.ground_truth.get("metadata", {}),
                    )
                    for predictions in file_predictions.boreholes
                ],
            )
        )

    @staticmethod
    def evaluate_gw(file_predictions: FilePredictionsWithGroundTruth) -> OverallGroundwaterMetrics:
        return GroundwaterEvaluator.evaluate(
            file_predictions=FileGroundwaterWithGroundTruth(
                file_predictions.filename,
                [
                    BoreholeGroundwaterWithGroundTruth(
                        predictions.predictions.groundwater_in_borehole if predictions.predictions else None,
                        predictions.ground_truth.get("groundwater", []) or [],  # value can be `None`
                    )
                    for predictions in file_predictions.boreholes
                ],
            )
        )

    @staticmethod
    def match_overall_with_ground_truth(
        overall_predictions: OverallFilePredictions, ground_truth: GroundTruth
    ) -> AllBoreholePredictionsWithGroundTruth:
        """Match all file predictions across multiple files with their corresponding ground truth data.

        Iterates over each file in overall_predictions and delegates per-file matching to
        LayerEvaluator. Files with no matching ground truth entry are included with an empty
        borehole list.

        Args:
            overall_predictions (OverallFilePredictions): Predictions for all processed files.
            ground_truth (GroundTruth): The ground truth dataset to match against.

        Returns:
            AllBoreholePredictionsWithGroundTruth: Predictions for every file, each paired with
                its corresponding ground truth boreholes.
        """
        return AllBoreholePredictionsWithGroundTruth(
            [
                Evaluator.match_with_ground_truth(file_predictions=file, ground_truth=ground_truth)
                for file in overall_predictions.file_predictions_list
            ]
        )

    @staticmethod
    def evaluate_overall(
        overall_predictions: AllBoreholePredictionsWithGroundTruth,
    ) -> tuple[OverallMetricsCatalog, OverallBoreholeMetadataMetrics]:
        fp_languages = {fp.filename: fp.language for fp in overall_predictions.predictions_list}
        languages = set(fp_languages.values())

        overall_groundwater_metrics = OverallGroundwaterMetrics()
        overall_metadata_metrics = OverallBoreholeMetadataMetrics()

        overall_geology_metrics = OverallMetricsCatalog(languages=languages)
        overall_layer_metrics = OverallMetrics()
        overall_depth_interval_metrics = OverallMetrics()
        overall_material_description_metrics = OverallMetrics()

        # iteration over all the file
        for file_predictions in overall_predictions.predictions_list:
            (
                overall_layer_metrics.metrics[file_predictions.filename],
                overall_depth_interval_metrics.metrics[file_predictions.filename],
                overall_material_description_metrics.metrics[file_predictions.filename],
            ) = Evaluator.evaluate_layers(file_predictions)
            gw_metrics = Evaluator.evaluate_gw(file_predictions)
            metadata_metrics = Evaluator.evaluate_metadata(file_predictions)

            overall_groundwater_metrics.add_groundwater_metrics(gw_metrics)
            overall_metadata_metrics.add_metadata_metrics(metadata_metrics)

        # Set metrics for geology
        overall_geology_metrics.layer_metrics = overall_layer_metrics
        overall_geology_metrics.depth_interval_metrics = overall_depth_interval_metrics
        overall_geology_metrics.material_description_metrics = overall_material_description_metrics
        overall_geology_metrics.groundwater_metrics = (
            overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        )
        overall_geology_metrics.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )

        # Set metrics for language
        def get_language_metric(fp: dict[str, Metrics], language: str) -> OverallMetrics:
            return OverallMetrics(
                {filename: metric for filename, metric in fp.items() if fp_languages.get(filename) == language}
            )

        for language in languages:
            setattr(
                overall_geology_metrics,
                f"{language}_layer_metrics",
                get_language_metric(overall_layer_metrics.metrics, language),
            )

            setattr(
                overall_geology_metrics,
                f"{language}_depth_interval_metrics",
                get_language_metric(overall_depth_interval_metrics.metrics, language),
            )

            setattr(
                overall_geology_metrics,
                f"{language}_material_description_metrics",
                get_language_metric(overall_material_description_metrics.metrics, language),
            )

        return overall_geology_metrics, overall_metadata_metrics
