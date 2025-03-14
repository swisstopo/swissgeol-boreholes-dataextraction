"""Classes for predictions per PDF file."""

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.metadata.metadata import FileMetadata
from stratigraphy.util.borehole_predictions import (
    BoreholePredictions,
    FilePredictionsWithGroundTruth,
)
from stratigraphy.util.file_predictions import FilePredictions
from stratigraphy.util.predictions import AllBoreholePredictionsWithGroundTruth


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
            "_".join([file_prediction.file_name, str(borehole_prediction.borehole_index)]): {
                "file_metadata": file_prediction.file_metadata.to_json(),
                "borehole_metadata": borehole_prediction.metadata.to_json(),
            }
            for file_prediction in self.file_predictions_list
            for borehole_prediction in file_prediction.borehole_predictions_list
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
            file_metadata = FileMetadata.from_json(file_data, file_name)

            borehole_list = [BoreholePredictions.from_json(bh_data, file_name) for bh_data in file_data["boreholes"]]

            overall_file_predictions.add_file_predictions(FilePredictions(borehole_list, file_metadata, file_name))
        return overall_file_predictions

    def match_with_ground_truth(self, ground_truth: GroundTruth) -> AllBoreholePredictionsWithGroundTruth:
        """Match the extracted boreholes with corresponding boreholes in the ground truth data.

        This is done by comparing the layers of the extracted boreholes with those in the groundtruth.

        Args:
            ground_truth (GroundTruth): The ground truth.

        Returns:
            AllBoreholePredictionsWithGroundTruth: all predictions per borehole with associated ground truth data.
        """
        files = []
        for file_predictions in self.file_predictions_list:
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
