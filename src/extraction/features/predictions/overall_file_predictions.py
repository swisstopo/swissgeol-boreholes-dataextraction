"""Classes for predictions per PDF file."""

import dataclasses

from extraction.features.predictions.file_predictions import FilePredictionsWithMetrics


@dataclasses.dataclass
class OverallFilePredictions:
    """A class to represent predictions for all files."""

    file_predictions_list: list[FilePredictionsWithMetrics] = dataclasses.field(default_factory=list)

    def contains(self, filename: str) -> bool:
        """Check if `file_predictions_list` contains `filename`.

        Args:
            filename (str): Filename to check.

        Returns:
            bool: True if `file_predictions_list` contains `filename`, else False.
        """
        return any(file.filename == filename for file in self.file_predictions_list)

    def add_file_predictions(self, file_predictions: FilePredictionsWithMetrics) -> None:
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
            "_".join([file_prediction.filename, str(borehole_prediction.borehole_index)]): {
                # TODO Add metadata
                # "file_metadata": file_prediction.file_metadata.to_json(),
                "borehole_metadata": borehole_prediction.metadata.to_json(),
            }
            for file_prediction in self.file_predictions_list
            for borehole_prediction in file_prediction.boreholes
        }

    def to_json(self) -> dict:
        """Converts the object to a dictionary by merging individual file predictions.

        Returns:
            dict: A dictionary representation of the object.
        """
        return {fp.filename: fp.to_json() for fp in self.file_predictions_list}

    @classmethod
    def from_json(cls, prediction_from_file: dict) -> "OverallFilePredictions":
        """Converts a dictionary to an object.

        Args:
            prediction_from_file (dict): A dictionary representing the predictions.

        Returns:
            OverallFilePredictions: The object.
        """
        overall_file_predictions = OverallFilePredictions()
        for filename, file_data in prediction_from_file.items():
            overall_file_predictions.add_file_predictions(FilePredictionsWithMetrics.from_json(file_data, filename))
        return overall_file_predictions
