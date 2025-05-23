"""Classes for predictions per PDF file."""

import dataclasses

from extraction.features.metadata.metadata import FileMetadata
from extraction.features.predictions.borehole_predictions import BoreholePredictions


@dataclasses.dataclass
class FilePredictions:
    """A class to represent predictions for a single file."""

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
