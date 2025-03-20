"""Classes for predictions per PDF file."""

import dataclasses
import io
import csv

from stratigraphy.metadata.metadata import FileMetadata
from stratigraphy.util.borehole_predictions import BoreholePredictions


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

    def to_csv(self) -> dict:
        """Converts the borehole predictions to a a list of CSV format strings.
        This method iterates through the list of borehole predictions and writes
        the relevant data (layer index, material description, start depth,
        and end depth) to a CSV string.

        Returns:
            list: a list of CSV string representation of the borehole predictions.
        """

        csv_strings = []

        for borehole in self.borehole_predictions_list:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["layer_index", "from_depth", "to_depth", "material_description"])

            for layer_index, layer in enumerate(borehole.layers_in_borehole.layers):
                start_depth = None if layer.depths is None or layer.depths.start is None else layer.depths.start.value
                end_depth = None if layer.depths is None or layer.depths.end is None else layer.depths.end.value
                material_description = None if layer.material_description is None else layer.material_description.feature.text

                writer.writerow([
                    layer_index,
                    start_depth,
                    end_depth,
                    material_description,
                ])

            csv_strings.append(output.getvalue())

        return csv_strings
