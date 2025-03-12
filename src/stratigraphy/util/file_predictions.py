"""Classes for predictions per PDF file."""

import dataclasses

from stratigraphy.metadata.metadata import FileMetadata
from stratigraphy.util.borehole_predictions import BoreholePredictions


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
