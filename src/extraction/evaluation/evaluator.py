"""Evaluator for borehole predictions against ground truth data."""

from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics, OverallBoreholeMetadataMetrics
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.evaluation.metadata_evaluator import MetadataEvaluator
from extraction.features.predictions.borehole_predictions import (
    BoreholeMetadataWithGroundTruth,
    FileMetadataWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.predictions.predictions import AllBoreholePredictionsWithGroundTruth


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
            boreholes = LayerEvaluator.match_predictions_with_ground_truth(file_predictions, ground_truth_for_file)
        return FilePredictionsWithGroundTruth(
            file_predictions.file_name, file_predictions.file_metadata.language, boreholes
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
        files = []
        for file_predictions in overall_predictions.file_predictions_list:
            boreholes = []
            ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
            if ground_truth_for_file:
                boreholes = LayerEvaluator.match_predictions_with_ground_truth(file_predictions, ground_truth_for_file)
            files.append(
                FilePredictionsWithGroundTruth(
                    file_predictions.file_name, file_predictions.file_metadata.language, boreholes
                )
            )
        return AllBoreholePredictionsWithGroundTruth(files)

    @staticmethod
    def evaluate_metadata_extraction(
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
    def evaluate_overall_metadata_extraction(
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
                Evaluator.evaluate_metadata_extraction(file_metadata_gt)
                for file_metadata_gt in overall_predictions.predictions_list
            ]
        )
