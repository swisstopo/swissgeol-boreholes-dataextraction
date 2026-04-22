"""Evaluator for borehole predictions against ground truth data."""

import logging

from core.benchmark_utils import Metrics
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.benchmark.metrics import OverallMetricsCatalog
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import (
    GroundwaterEvaluator,
    GroundwaterMetrics,
    OverallGroundwaterMetrics,
)
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

logger = logging.getLogger(__name__)


class Evaluator:
    """Static utility class for matching borehole predictions with ground truth and computing evaluation metrics."""

    @staticmethod
    def match_with_ground_truth(
        file_predictions: FilePredictions, ground_truth: GroundTruth
    ) -> FilePredictionsWithGroundTruth:
        """Match the extracted boreholes with corresponding boreholes in the ground truth data.

        This is done by comparing the layers of the extracted boreholes with those in the ground truth.

        Args:
            file_predictions (FilePredictions): File predictions.
            ground_truth (GroundTruth): The ground truth.

        Returns:
            FilePredictionsWithGroundTruth: All predictions per borehole with associated ground truth data.
        """
        boreholes = []
        ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
        if ground_truth_for_file:
            boreholes = LayerEvaluator.match_boreholes_to_ground_truth(file_predictions, ground_truth_for_file)
        return FilePredictionsWithGroundTruth(
            file_predictions.file_name, file_predictions.file_metadata.language, boreholes
        )

    @staticmethod
    def evaluate(
        file_predictions: FilePredictionsWithGroundTruth,
    ) -> tuple[Metrics, Metrics, Metrics, GroundwaterMetrics, BoreholeMetadataMetrics]:
        """Evaluate layers, metadata, and groundwater for a single file prediction.

        Sets the `is_correct` flags on layers, depth intervals, material descriptions, groundwater entries,
        and metadata fields inside `file_predictions`.

        Args:
            file_predictions (FilePredictionsWithGroundTruth): Per-file borehole predictions
                paired with their ground truth data.

        Returns:
            tuple[Metrics, Metrics, Metrics, GroundwaterMetrics, BoreholeMetadataMetrics]:
                A 5-tuple of:
                - layer_metrics (Metrics): Metrics for layer detection.
                - depth_interval_metrics (Metrics): Metrics for depth intervals.
                - material_description_metrics (Metrics): Metrics for material descriptions.
                - groundwater_metrics (GroundwaterMetrics): Metrics for Groundwater detection.
                - metadata_metrics (BoreholeMetadataMetrics): Metrics for elevation, coordinate, and name.
        """
        layer_metrics, depth_interval_metrics, material_description_metrics = Evaluator._evaluate_layers(
            file_predictions
        )
        gw_metrics = Evaluator._evaluate_gw(file_predictions)
        metadata_metrics = Evaluator._evaluate_metadata(file_predictions)

        return layer_metrics, depth_interval_metrics, material_description_metrics, gw_metrics, metadata_metrics

    @staticmethod
    def _evaluate_layers(file_predictions: FilePredictionsWithGroundTruth) -> tuple[Metrics, Metrics, Metrics]:
        """Evaluate layer, depth-interval, and material-description metrics for a single file.

        Args:
            file_predictions (FilePredictionsWithGroundTruth): Per-file borehole predictions
                paired with their ground truth data.

        Returns:
            tuple[Metrics, Metrics, Metrics]: (layer_metrics, depth_interval_metrics, material_description_metrics)
        """
        return LayerEvaluator.evaluate(
            file_predictions=FileLayersWithGroundTruth(
                file_predictions.filename,
                file_predictions.language,
                [
                    BoreholeLayersWithGroundTruth(
                        borehole.predictions.layers_in_borehole if borehole.predictions else None,
                        borehole.ground_truth.get("layers", []),
                    )
                    for borehole in file_predictions.boreholes
                ],
            )
        )

    @staticmethod
    def _evaluate_metadata(
        file_predictions: FilePredictionsWithGroundTruth,
    ) -> BoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            file_predictions (FilePredictionsWithGroundTruth): Per-file borehole predictions
                paired with their ground truth data.

        Returns:
            BoreholeMetadataMetrics: The computed metrics for the metadata.
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
    def _evaluate_gw(file_predictions: FilePredictionsWithGroundTruth) -> GroundwaterMetrics:
        """Evaluate groundwater metrics for a single file prediction.

        Args:
            file_predictions (FilePredictionsWithGroundTruth): Per-file borehole predictions
                paired with their ground truth data.

        Returns:
            GroundwaterMetrics: The computed groundwater metrics for the file.
        """
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
    def aggregate(
        overall_predictions: OverallFilePredictions,
    ) -> OverallMetricsCatalog:
        """Aggregate all files prediction for geology and metadata metrics.

        Args:
            overall_predictions (OverallFilePredictions): All per-file predictions
                paired with their ground truth boreholes.

        Returns:
            tuple[OverallMetricsCatalog]:
                - OverallMetricsCatalog: aggregated geology metrics (layers, depth intervals,
                  material descriptions, groundwater) globally and per language.
                - OverallBoreholeMetadataMetrics: aggregated metadata metrics across all files.
        """
        fp_languages = {fp.filename: fp.language for fp in overall_predictions.file_predictions_list}
        languages = set(fp_languages.values())

        overall_metrics_catalog = OverallMetricsCatalog(languages=languages)
        overall_groundwater_metrics = OverallGroundwaterMetrics()

        # Iterate over all the files
        for predictions in overall_predictions.file_predictions_list:
            overall_metrics_catalog.add_datapoint(
                filename=predictions.filename,
                material_description_metric=predictions.metrics.material_description_metrics,
                layer_metrics=predictions.metrics.layer_metrics,
                depth_interval_metric=predictions.metrics.depth_interval_metrics,
                elevation_metric=predictions.metrics.metadata_metrics.elevation_metrics,
                coordinates_metric=predictions.metrics.metadata_metrics.coordinates_metrics,
                name_metric=predictions.metrics.metadata_metrics.name_metrics,
            )

            # Assign values to overall predictions
            overall_groundwater_metrics.add_groundwater_metrics(predictions.metrics.gw_metrics)

        # Set metrics for geology
        overall_metrics_catalog.groundwater_metrics = (
            overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        )
        overall_metrics_catalog.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )

        # Language subsets for geology
        for language in languages:
            setattr(
                overall_metrics_catalog,
                f"{language}_layer_metrics",
                overall_metrics_catalog.layer_metrics.get_language_subset(fp_languages, language),
            )

            setattr(
                overall_metrics_catalog,
                f"{language}_depth_interval_metrics",
                overall_metrics_catalog.depth_interval_metrics.get_language_subset(fp_languages, language),
            )

            setattr(
                overall_metrics_catalog,
                f"{language}_material_description_metrics",
                overall_metrics_catalog.material_description_metrics.get_language_subset(fp_languages, language),
            )

        return overall_metrics_catalog
