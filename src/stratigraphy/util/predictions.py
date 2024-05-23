"""This module contains classes for predictions."""

import contextlib
import logging
import math
import uuid
from collections import defaultdict
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


@dataclass
class PagePredictions:
    """A class to represent predictions for a single page."""

    layers: list[LayerPrediction]
    page_number: int
    page_width: int
    page_height: int
    depths_materials_columns_pairs: list[dict] = None

    def __post__init__(self):
        """Sort layers by their occurence on the page."""
        self.layers = sorted(self.layers, key=lambda layer: layer.material_description.rect.y0)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(self, pages: list[PagePredictions], file_name: str, language: str, metadata: BoreholeMetaData = None):
        self.pages = pages
        self.file_name = file_name
        self.language = language
        self.layers = sum([page.layers for page in self.pages], [])
        self.metadata = metadata
        self.metadata_is_correct: dict = {}

    @staticmethod
    def create_from_json(predictions_for_file: dict, file_name: str):
        """Create predictions class for a file given the predictions.json file.

        Args:
            predictions_for_file (dict): The predictions for the file in json format.
            file_name (str): The name of the file.
        """
        page_predictions_class = []
        for page_number, page_predictions in predictions_for_file.items():
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
            page_layers = page_predictions["layers"]
            layer_predictions = []
            for layer in page_layers:
                material_prediction = _create_textblock_object(layer["material_description"]["lines"])
                if "depth_interval" in layer:
                    start = (
                        DepthColumnEntry(
                            value=layer["depth_interval"]["start"]["value"],
                            rect=fitz.Rect(layer["depth_interval"]["start"]["rect"]),
                        )
                        if layer["depth_interval"]["start"] is not None
                        else None
                    )
                    end = (
                        DepthColumnEntry(
                            value=layer["depth_interval"]["end"]["value"],
                            rect=fitz.Rect(layer["depth_interval"]["end"]["rect"]),
                        )
                        if layer["depth_interval"]["end"] is not None
                        else None
                    )

                    depth_interval_prediction = BoundaryInterval(start=start, end=end)
                    layer_predictions.append(
                        LayerPrediction(
                            material_description=material_prediction, depth_interval=depth_interval_prediction
                        )
                    )
                else:
                    layer_predictions.append(
                        LayerPrediction(material_description=material_prediction, depth_interval=None)
                    )
            if "depths_materials_column_pairs" in page_predictions:
                page_predictions_class.append(
                    PagePredictions(
                        page_number=page_number,
                        page_width=page_predictions["page_width"],
                        page_height=page_predictions["page_height"],
                        layers=layer_predictions,
                        depths_materials_columns_pairs=page_predictions["depths_materials_column_pairs"],
                    )
                )
            else:
                page_predictions_class.append(PagePredictions(page_number=page_number, layers=layer_predictions))

        return FilePredictions(
            pages=page_predictions_class, file_name=file_name, language=file_language, metadata=file_metadata
        )

    @staticmethod
    def create_from_label_studio(annotation_results: dict):
        """Create predictions class for a file given the annotation results from Label Studio.

        This method is meant to import annotations from label studio. The primary use case is to
        use the annotated data for evaluation. For that purpose, there is the convert_to_ground_truth
        method, which then converts the predictions to ground truth format.

        NOTE: We may want to adjust this method to return a single instance of the class,
        instead of a list of class objects.

        NOTE: Before using this to create new ground truth, this method should be tested.

        Args:
            annotation_results (dict): The annotation results from Label Studio.
                                       The annotation_results can cover multiple files.

        Returns:
            list[FilePredictions]: A list of FilePredictions objects, one for each file present in the
                                   annotation_results.
        """
        file_pages = defaultdict(list)
        for annotation in annotation_results:
            # get page level information
            file_name, page_index = _get_file_name_and_page_index(annotation)
            page_width = annotation["annotations"][0]["result"][0]["original_width"]
            page_height = annotation["annotations"][0]["result"][0]["original_height"]

            # extract all material descriptions and depth intervals and link them together
            # Note: we need to loop through the annotations twice, because the order of the annotations is
            # not guaranteed. In the first iteration we grasp all IDs, in the second iteration we extract the
            # information for each id.
            material_descriptions = {}
            depth_intervals = {}
            linking_objects = []

            # define all the material descriptions and depth intervals with their ids
            for annotation_result in annotation["annotations"][0]["result"]:
                if annotation_result["type"] == "labels":
                    if annotation_result["value"]["labels"] == ["Material Description"]:
                        material_descriptions[annotation_result["id"]] = {
                            "rect": annotation_result["value"]
                        }  # TODO extract rectangle properly
                    elif annotation_result["value"]["labels"] == ["Depth Interval"]:
                        depth_intervals[annotation_result["id"]] = {}
                if annotation_result["type"] == "relation":
                    linking_objects.append(
                        {"from_id": annotation_result["from_id"], "to_id": annotation_result["to_id"]}
                    )

            # check annotation results for material description or depth interval ids
            for annotation_result in annotation["annotations"][0]["result"]:
                with contextlib.suppress(KeyError):
                    id = annotation_result["id"]  # relation regions do not have an ID.
                if annotation_result["type"] == "textarea":
                    if id in material_descriptions:
                        material_descriptions[id]["text"] = annotation_result["value"]["text"][
                            0
                        ]  # There is always only one element. TO CHECK!
                        if len(annotation_result["value"]["text"]) > 1:
                            print(f"More than one text in material description: {annotation_result['value']['text']}")
                    elif id in depth_intervals:
                        depth_interval_text = annotation_result["value"]["text"][0]
                        start, end = _get_start_end_from_text(depth_interval_text)
                        depth_intervals[id]["start"] = start
                        depth_intervals[id]["end"] = end
                        depth_intervals[id]["background_rect"] = annotation_result[
                            "value"
                        ]  # TODO extract rectangle properly
                    else:
                        print(f"Unknown id: {id}")

            # create the layer prediction objects by linking material descriptions with depth intervals
            layers = []

            for link in linking_objects:
                from_id = link["from_id"]
                to_id = link["to_id"]
                material_description_prediction = MaterialDescription(**material_descriptions.pop(from_id))
                depth_interval_prediction = AnnotatedInterval(**depth_intervals.pop(to_id))
                layers.append(
                    LayerPrediction(
                        material_description=material_description_prediction,
                        depth_interval=depth_interval_prediction,
                        material_is_correct=True,
                        depth_interval_is_correct=True,
                    )
                )

            if material_descriptions or depth_intervals:
                # TODO: This should not be acceptable. Raising an error doesnt seem the right way to go either.
                # But at least it should be warned.
                print("There are material descriptions or depth intervals left over.")
                print(material_descriptions)
                print(depth_intervals)

            file_pages[file_name].append(
                PagePredictions(layers=layers, page_number=page_index, page_width=page_width, page_height=page_height)
            )

        file_predictions = []
        for file_name, page_predictions in file_pages.items():
            file_predictions.append(
                FilePredictions(file_name=file_name, pages=page_predictions, language="unknown")
            )  # TODO: language should not be required here.

        return file_predictions

    def convert_to_ground_truth(self):
        """Convert the predictions to ground truth format.

        This method is meant to be used in combination with the create_from_label_studio method.
        It converts the predictions to ground truth format, which can then be used for evaluation.

        NOTE: This method should be tested before using it to create new ground truth.

        Returns:
            dict: The predictions in ground truth format.
        """
        for page in self.pages:
            layers = []
            for layer in page.layers:
                material_description = layer.material_description.text
                depth_interval = {
                    "start": layer.depth_interval.start.value if layer.depth_interval.start else None,
                    "end": layer.depth_interval.end.value if layer.depth_interval.end else None,
                }
                layers.append({"material_description": material_description, "depth_interval": depth_interval})
            ground_truth = {self.file_name: {"layers": layers}}
        return ground_truth

    def evaluate(self, ground_truth: dict):
        self.evaluate_layers(ground_truth["layers"])
        self.evaluate_metadata(ground_truth.get("metadata"))

    def evaluate_layers(self, ground_truth_layers: list):
        """Evaluate all layers of the predictions against the ground truth.

        Args:
            ground_truth_layers (list): The ground truth layers for the file.
        """
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
                ground_truth_west = int(metadata_ground_truth["coordinates"]["N"]) + 1e6
            elif (
                self.metadata.coordinates.east.coordinate_value < 2e6
                and metadata_ground_truth["coordinates"]["E"] > 2e6
            ):
                ground_truth_east = int(metadata_ground_truth["coordinates"]["E"]) - 2e6
                ground_truth_west = int(metadata_ground_truth["coordinates"]["N"]) - 1e6
            else:
                ground_truth_east = int(metadata_ground_truth["coordinates"]["E"])
                ground_truth_west = int(metadata_ground_truth["coordinates"]["N"])

            if (
                math.isclose(int(self.metadata.coordinates.east.coordinate_value), ground_truth_east, rel_tol=0.001)
            ) and (
                math.isclose(int(self.metadata.coordinates.north.coordinate_value), ground_truth_west, rel_tol=0.001)
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


def _get_start_end_from_text(text: str) -> tuple[float]:
    start, end = text.split("end: ")
    start = start.split("start: ")[1]
    return float(start), float(end)


def _get_file_name_and_page_index(annotation):
    file_name = annotation["data"]["ocr"].split("/")[-1]
    file_name = file_name.split(".")[0]
    return file_name.split("_")
