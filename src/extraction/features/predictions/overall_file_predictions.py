"""Classes for predictions per PDF file."""

from pydantic import BaseModel, Field, model_serializer

from extraction.features.predictions.file_predictions import FilePredictionsWithMetrics


class OverallFilePredictions(BaseModel):
    """A class to represent predictions for all files."""

    file_predictions_list: list[FilePredictionsWithMetrics] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_flat_json(self, handler) -> dict:
        """Serialize as a dict with the filename as key (assumed ot be unique)."""
        serialized_dict = handler(self)

        return {entry.pop("filename"): entry for entry in serialized_dict["file_predictions_list"]}

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
