"""TODO."""

from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.features.predictions.borehole_predictions import (
    FilePredictionsWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions


class Evaluator:
    """TODO."""

    def match_with_ground_truth(
        file_predictions: FilePredictions, ground_truth: GroundTruth
    ) -> FilePredictionsWithGroundTruth:
        """Match the extracted boreholes with corresponding boreholes in the ground truth data.

        This is done by comparing the layers of the extracted boreholes with those in the groundtruth.

        Args:
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

    # TODO: hello
    # def evaluate_metadata_extraction(self) -> OverallBoreholeMetadataMetrics:
    #     """Evaluate the metadata extraction of the predictions against the ground truth.

    #     Returns:
    #         OverallBoreholeMetadataMetrics: the computed metrics for the metadata.
    #     """
    #     metadata_list = [
    #         FileMetadataWithGroundTruth(
    #             file.filename,
    #             [
    #                 BoreholeMetadataWithGroundTruth(
    #                     predictions.predictions.metadata if predictions.predictions else None,
    #                     predictions.ground_truth.get("metadata", {}),
    #                 )
    #                 for predictions in file.boreholes
    #             ],
    #         )
    #         for file in self.predictions_list
    #     ]

    #     return MetadataEvaluator(metadata_list).evaluate()
