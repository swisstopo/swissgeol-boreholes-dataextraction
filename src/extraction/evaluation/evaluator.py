"""Evaluator for borehole predictions against ground truth data."""

import logging

from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics, OverallBoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterEvaluator, OverallGroundwaterMetrics
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.evaluation.metadata_evaluator import MetadataEvaluator
from extraction.features.predictions.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    BoreholeMetadataWithGroundTruth,
    FileGroundwaterWithGroundTruth,
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
        LayerEvaluator.evaluate(file_predictions)

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
        file_metadata_gt = FileMetadataWithGroundTruth(
            file_predictions.filename,
            [
                BoreholeMetadataWithGroundTruth(
                    predictions.predictions.metadata if predictions.predictions else None,
                    predictions.ground_truth.get("metadata", {}),
                )
                for predictions in file_predictions.boreholes
            ],
        )

        return MetadataEvaluator.evaluate(file_metadata_gt)

    @staticmethod
    def evaluate_gw(file_predictions: FilePredictionsWithGroundTruth) -> OverallGroundwaterMetrics:
        groundwater = FileGroundwaterWithGroundTruth(
            file_predictions.filename,
            [
                BoreholeGroundwaterWithGroundTruth(
                    predictions.predictions.groundwater_in_borehole if predictions.predictions else None,
                    predictions.ground_truth.get("groundwater", []) or [],  # value can be `None`
                )
                for predictions in file_predictions.boreholes
            ],
        )

        return GroundwaterEvaluator.evaluate(groundwater)

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
    def evaluate_overall(overall_predictions: AllBoreholePredictionsWithGroundTruth) -> tuple[BoreholeMetadataMetrics]:
        Evaluator.evaluate_overall_layer(overall_predictions)
        metadata_metrics_list = Evaluator.evaluate_overall_metadata(overall_predictions)
        Evaluator.evaluate_overall_gw(overall_predictions)
        return metadata_metrics_list

    @staticmethod
    def evaluate_overall_layer(overall_predictions: AllBoreholePredictionsWithGroundTruth):
        for file_predictions in overall_predictions.predictions_list:
            LayerEvaluator.evaluate(file_predictions)

    # @staticmethod
    # def evaluate_overall_geology(overall_predictions: AllBoreholePredictionsWithGroundTruth) ->OverallMetricsCatalog:
    #     """Evaluate the borehole extraction predictions.

    #     Args:
    #         overall_predictions (AllBoreholePredictionsWithGroundTruth): metrics for the current evaluation.

    #     Returns:
    #         OverallMetricsCatalog: A OverallMetricsCatalog that maps a metrics name to the corresponding
    #         OverallMetrics object. If no ground truth is available, None is returned.
    #     """
    #     languages = set(fp.language for fp in overall_predictions.predictions_list)
    #     all_metrics = OverallMetricsCatalog(languages=languages)

    #     layers_list = [
    #         FileLayersWithGroundTruth(
    #             file.filename,
    #             file.language,
    #             [
    #                 BoreholeLayersWithGroundTruth(
    #                     predictions.predictions.layers_in_borehole if predictions.predictions else None,
    #                     predictions.ground_truth.get("layers", []),
    #                 )
    #                 for predictions in file.boreholes
    #             ],
    #         )
    #         for file in overall_predictions.predictions_list
    #     ]
    #     evaluator = LayerEvaluator(layers_list)
    #     all_metrics.material_description_metrics = evaluator.get_material_description_metrics()
    #     all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()
    #     all_metrics.layer_metrics = evaluator.get_layer_metrics()

    #     predictions_by_language = {language: [] for language in languages}
    #     for borehole_data in layers_list:
    #         # even if metadata can be different for boreholes in the same document,
    #         # langage is the same (take index 0)
    #         predictions_by_language[borehole_data.language].append(borehole_data)

    #     for language, language_predictions_list in predictions_by_language.items():
    #         evaluator = LayerEvaluator(language_predictions_list)
    #         setattr(all_metrics, f"{language}_layer_metrics", evaluator.get_layer_metrics())
    #         setattr(all_metrics, f"{language}_depth_interval_metrics", evaluator.get_depth_interval_metrics())
    #         setattr(
    #             all_metrics, f"{language}_material_description_metrics", evaluator.get_material_description_metrics()
    #         )

    #     logger.info("Macro avg:")
    #     logger.info(
    #         "layer f1: %.1f%%, depth interval f1: %.1f%%, material description f1: %.1f%%",
    #         all_metrics.layer_metrics.macro_f1() * 100,
    #         all_metrics.depth_interval_metrics.macro_f1() * 100,
    #         all_metrics.material_description_metrics.macro_f1() * 100,
    #     )

    #     # TODO groundwater should not be in evaluate_geology(), it should be handle by a higher-level function call
    #     groundwater_list = [
    #         FileGroundwaterWithGroundTruth(
    #             file.filename,
    #             [
    #                 BoreholeGroundwaterWithGroundTruth(
    #                     predictions.predictions.groundwater_in_borehole if predictions.predictions else None,
    #                     predictions.ground_truth.get("groundwater", []) or [],  # value can be `None`
    #                 )
    #                 for predictions in file.boreholes
    #             ],
    #         )
    #         for file in overall_predictions.predictions_list
    #     ]
    #     overall_groundwater_metrics = GroundwaterEvaluator(groundwater_list).evaluate()
    #     all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
    #     all_metrics.groundwater_depth_metrics = (
    #         overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
    #     )
    #     return all_metrics

    @staticmethod
    def evaluate_overall_metadata(
        overall_predictions: AllBoreholePredictionsWithGroundTruth,
    ) -> OverallBoreholeMetadataMetrics:
        """Evaluate metadata extraction for all files in the overall predictions.

        Delegates per-file evaluation to evaluate_metadata_extraction and collects the results.

        Args:
            overall_predictions (AllBoreholePredictionsWithGroundTruth): All per-file borehole
                predictions paired with their ground truth data.

        Returns:
            OverallBoreholeMetadataMetrics: Aggregated metadata metrics across all files.
        """
        return OverallBoreholeMetadataMetrics(
            [
                Evaluator.evaluate_metadata(file_metadata_gt)
                for file_metadata_gt in overall_predictions.predictions_list
            ]
        )

    @staticmethod
    def evaluate_overall_gw(overall_predictions: AllBoreholePredictionsWithGroundTruth) -> OverallGroundwaterMetrics:
        return OverallGroundwaterMetrics(
            [Evaluator.evaluate_gw(file_predictions) for file_predictions in overall_predictions.predictions_list]
        )
