"""Classes for predictions per PDF file."""

from pydantic import BaseModel, Field

from extraction.features.predictions.file_predictions import FilePredictionsWithMetrics


class OverallFilePredictions(BaseModel):
    """A class to represent predictions for all files."""

    file_predictions_list: list[FilePredictionsWithMetrics] = Field(default_factory=list)

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
            file_predictions (FilePredictionsWithMetrics): The file predictions to add.
        """
        self.file_predictions_list.append(file_predictions)
