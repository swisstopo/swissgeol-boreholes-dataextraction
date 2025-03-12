"""Classes for predictions per PDF file."""

import dataclasses

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.metadata.metadata import FileMetadata
from stratigraphy.util.borehole_predictions import (
    BoreholePredictions,
    BoreholePredictionsWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from stratigraphy.util.predictions import AllBoreholePredictionsWithGroundTruth


@dataclasses.dataclass
class FilePredictions:
    """A class to represent predictions for a single file.

    It is responsible for grouping all the lists of elements into a single list of BoreholePrediction objects.
    """

    borehole_predictions_list: list[BoreholePredictions]
    file_metadata: FileMetadata
    file_name: str

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            **self.file_metadata.to_json(),
            "boreholes": [borehole.to_json() for borehole in self.borehole_predictions_list],
        }


class OverallFilePredictions:
    """A class to represent predictions for all files."""

    def __init__(self) -> None:
        """Initializes the OverallFilePredictions object."""
        self.file_predictions_list: list[FilePredictions] = []
        self.matching_pred_to_gt_boreholes = dict()  # set when evaluating layers

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

    ############################################################################################################
    ### Evaluation methods
    ############################################################################################################

    def match_with_ground_truth(self, ground_truth: GroundTruth) -> AllBoreholePredictionsWithGroundTruth:
        """Match the extracted boreholes with corresponding boreholes in the ground truth data.

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
                borehole_matching_gt_to_pred = LayerEvaluator.evaluate_borehole(
                    [bh.layers_in_borehole for bh in file_predictions.borehole_predictions_list],
                    {idx: borehole_data["layers"] for idx, borehole_data in ground_truth_for_file.items()},
                )
                for gt_idx, borehole_idx in borehole_matching_gt_to_pred.items():
                    boreholes.append(
                        BoreholePredictionsWithGroundTruth(
                            predictions=file_predictions.borehole_predictions_list[borehole_idx],
                            ground_truth=ground_truth_for_file[gt_idx],
                        )
                    )
            files.append(
                FilePredictionsWithGroundTruth(
                    file_predictions.file_name, file_predictions.file_metadata.language, boreholes
                )
            )
        return AllBoreholePredictionsWithGroundTruth(files)
