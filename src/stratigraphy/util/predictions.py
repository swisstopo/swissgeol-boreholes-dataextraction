"""This module contains classes for predictions."""

import logging
import math
import uuid
from dataclasses import dataclass, field

import fitz
import Levenshtein

from stratigraphy.util.coordinate_extraction import Coordinate
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.interval import AnnotatedInterval, BoundaryInterval
from stratigraphy.util.line import TextLine, TextWord
from stratigraphy.util.textblock import MaterialDescription, TextBlock
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


@dataclass
class BoreholeMetaData:
    """Class to represent metadata of a borehole profile."""

    coordinates: Coordinate | None


@dataclass
class LayerPrediction:
    """A class to represent predictions for a single layer."""

    material_description: TextBlock | MaterialDescription
    depth_interval: BoundaryInterval | AnnotatedInterval
    material_is_correct: bool = None
    depth_interval_is_correct: bool = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers: list[LayerPrediction],
        file_name: str,
        language: str,
        metadata: BoreholeMetaData = None,
        depths_materials_columns_pairs: list[dict] = None,
        page_sizes: list[tuple[int, int]] = None,
    ):
        self.layers: list[LayerPrediction] = sorted(layers, key=lambda layer: layer.material_description.rect.y0)
        self.depths_materials_columns_pairs: list[dict] = depths_materials_columns_pairs
        self.file_name = file_name
        self.language = language
        self.metadata = metadata
        self.metadata_is_correct: dict = {}
        self.page_sizes: list[tuple[int, int]] = page_sizes

    @staticmethod
    def create_from_json(predictions_for_file: dict, file_name: str):
        """Create predictions class for a file given the predictions.json file.

        Args:
            predictions_for_file (dict): The predictions for the file in json format.
            file_name (str): The name of the file.
        """
        page_layer_predictions_list: list[LayerPrediction] = []
        pages_width_list: list[int] = []
        pages_height_list: list[int] = []
        depths_materials_columns_pairs_list: list[dict] = []

        for page_number, page_predictions in predictions_for_file.items():
            # TODO: Look into this as it seems to be a quite dirty fix here.
            # As languages and metadata are not pages, they should be handled differently.
            if page_number == "language":
                file_language = page_predictions
                continue
            elif page_number == "metadata":
                metadata = page_predictions
                if "coordinates" in metadata:
                    if metadata["coordinates"] is not None:
                        coordinates = Coordinate.from_json(metadata["coordinates"])
                    else:
                        coordinates = None
                file_metadata = BoreholeMetaData(coordinates=coordinates)
                # TODO: Add additional metadata here.
                continue
            elif page_number == "layers":
                for layer in page_predictions:
                    material_prediction = _create_textblock_object(layer["material_description"]["lines"])
                    if "depth_interval" in layer:
                        start = (
                            DepthColumnEntry(
                                value=layer["depth_interval"]["start"]["value"],
                                rect=fitz.Rect(layer["depth_interval"]["start"]["rect"]),
                                page_number=layer["depth_interval"]["start"]["page"],
                            )
                            if layer["depth_interval"]["start"] is not None
                            else None
                        )
                        end = (
                            DepthColumnEntry(
                                value=layer["depth_interval"]["end"]["value"],
                                rect=fitz.Rect(layer["depth_interval"]["end"]["rect"]),
                                page_number=layer["depth_interval"]["end"]["page"],
                            )
                            if layer["depth_interval"]["end"] is not None
                            else None
                        )

                        depth_interval_prediction = BoundaryInterval(start=start, end=end)
                        layer_predictions = LayerPrediction(
                            material_description=material_prediction, depth_interval=depth_interval_prediction
                        )
                    else:
                        layer_predictions = LayerPrediction(
                            material_description=material_prediction, depth_interval=None
                        )

                    page_layer_predictions_list.append(layer_predictions)

            if "depths_materials_column_pairs" in page_predictions:
                depths_materials_columns_pairs_list.extend(page_predictions["depths_materials_column_pairs"])

            pages_width_list.extend(predictions_for_file["page_width"])
            pages_height_list.extend(predictions_for_file["page_height"])

        return FilePredictions(
            layers=page_layer_predictions_list,
            file_name=file_name,
            language=file_language,
            metadata=file_metadata,
            depths_materials_columns_pairs=depths_materials_columns_pairs_list,
            page_sizes=list(zip(pages_width_list, pages_height_list, strict=False)),
        )

    def convert_to_ground_truth(self):
        """Convert the predictions to ground truth format.

        This method is meant to be used in combination with the create_from_label_studio method.
        It converts the predictions to ground truth format, which can then be used for evaluation.

        NOTE: This method should be tested before using it to create new ground truth.

        Returns:
            dict: The predictions in ground truth format.
        """
        ground_truth = {self.file_name: {"metadata": self.metadata}}
        layers = []
        for layer in self.layers:
            material_description = layer.material_description.text
            depth_interval = {
                "start": layer.depth_interval.start.value if layer.depth_interval.start else None,
                "end": layer.depth_interval.end.value if layer.depth_interval.end else None,
            }
            layers.append({"material_description": material_description, "depth_interval": depth_interval})
        ground_truth[self.file_name]["layers"] = layers
        if self.metadata.coordinates is not None:
            ground_truth[self.file_name]["metadata"] = {
                "coordinates": {
                    "E": self.metadata.coordinates.east.coordinate_value,
                    "N": self.metadata.coordinates.north.coordinate_value,
                }
            }
        return ground_truth

    def evaluate(self, ground_truth: dict):
        """Evaluate the predictions against the ground truth.

        Args:
            ground_truth (dict): The ground truth for the file.
        """
        self.evaluate_layers(ground_truth["layers"])
        self.evaluate_metadata(ground_truth.get("metadata"))

    def evaluate_layers(self, ground_truth_layers: list):
        """Evaluate all layers of the predictions against the ground truth.

        Args:
            ground_truth_layers (list): The ground truth layers for the file.
        """
        # TODO: Attribute 'unmatched_layers' defined outside __init__ method. This is not a good practice.
        self.unmatched_layers = ground_truth_layers.copy()
        for layer in self.layers:
            match, depth_interval_is_correct = self._find_matching_layer(layer)
            if match:
                layer.material_is_correct = True
                layer.depth_interval_is_correct = depth_interval_is_correct
            else:
                layer.material_is_correct = False
                layer.depth_interval_is_correct = None

    def evaluate_metadata(self, metadata_ground_truth: dict):
        """Evaluate the metadata of the file against the ground truth.

        Note: For now coordinates is the only metadata extracted and evaluated for.

        Args:
            metadata_ground_truth (dict): The ground truth for the file.
        """
        if self.metadata.coordinates is None or (
            metadata_ground_truth is None or metadata_ground_truth.get("coordinates") is None
        ):
            self.metadata_is_correct["coordinates"] = None

        else:
            if (
                self.metadata.coordinates.east.coordinate_value > 2e6
                and metadata_ground_truth["coordinates"]["E"] < 2e6
            ):
                ground_truth_east = int(metadata_ground_truth["coordinates"]["E"]) + 2e6
                ground_truth_north = int(metadata_ground_truth["coordinates"]["N"]) + 1e6
            elif (
                self.metadata.coordinates.east.coordinate_value < 2e6
                and metadata_ground_truth["coordinates"]["E"] > 2e6
            ):
                ground_truth_east = int(metadata_ground_truth["coordinates"]["E"]) - 2e6
                ground_truth_north = int(metadata_ground_truth["coordinates"]["N"]) - 1e6
            else:
                ground_truth_east = int(metadata_ground_truth["coordinates"]["E"])
                ground_truth_north = int(metadata_ground_truth["coordinates"]["N"])

            if (math.isclose(int(self.metadata.coordinates.east.coordinate_value), ground_truth_east, abs_tol=2)) and (
                math.isclose(int(self.metadata.coordinates.north.coordinate_value), ground_truth_north, abs_tol=2)
            ):
                self.metadata_is_correct["coordinates"] = True
            else:
                self.metadata_is_correct["coordinates"] = False

    def _find_matching_layer(self, layer: LayerPrediction) -> tuple[dict, bool] | tuple[None, None]:
        """Find the matching layer in the ground truth.

        Args:
            layer (LayerPrediction): The layer to match.

        Returns:
            tuple[dict, bool] | tuple[None, None]: The matching layer and a boolean indicating if the depth interval
                                is correct. None if no match was found.
        """
        parsed_text = parse_text(layer.material_description.text)
        possible_matches = [
            ground_truth_layer
            for ground_truth_layer in self.unmatched_layers
            if Levenshtein.ratio(parsed_text, ground_truth_layer["material_description"]) > 0.9
        ]

        if not possible_matches:
            return None, None

        for possible_match in possible_matches:
            start = possible_match["depth_interval"]["start"]
            end = possible_match["depth_interval"]["end"]

            if layer.depth_interval is None:
                pass

            elif (
                start == 0 and layer.depth_interval.start is None and end == layer.depth_interval.end.value
            ):  # If not specified differently, we start at 0.
                self.unmatched_layers.remove(possible_match)
                return possible_match, True

            elif (  # noqa: SIM102
                layer.depth_interval.start is not None and layer.depth_interval.end is not None
            ):  # In all other cases we do not allow a None value.
                if start == layer.depth_interval.start.value and end == layer.depth_interval.end.value:
                    self.unmatched_layers.remove(possible_match)
                    return possible_match, True

        match = max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, x["material_description"]))
        self.unmatched_layers.remove(match)
        return match, False


def _create_textblock_object(lines: dict) -> TextBlock:
    lines = [TextLine([TextWord(**line)]) for line in lines]
    return TextBlock(lines)
