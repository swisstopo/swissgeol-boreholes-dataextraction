"""This module contains classes for predictions."""

import uuid
from dataclasses import dataclass

import fitz
import Levenshtein

from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.interval import BoundaryInterval
from stratigraphy.util.line import TextLine, TextWord
from stratigraphy.util.textblock import TextBlock
from stratigraphy.util.util import parse_text


@dataclass
class LayerPrediction:
    """A class to represent predictions for a single layer."""

    material_description: TextBlock
    depth_interval: BoundaryInterval
    material_is_correct: bool = None
    depth_interval_is_correct: bool = None
    id: str = uuid.uuid4().hex


@dataclass
class PagePredictions:
    """A class to represent predictions for a single page."""

    layers: list[LayerPrediction]
    page_number: int
    depths_materials_columns_pairs: list[dict] = None


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(self, pages: list[PagePredictions], file_name: str, language: str):
        self.pages = pages
        self.file_name = file_name
        self.language = language
        if self.pages:
            self.layers = sum([page.layers for page in self.pages], [])

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
                        layers=layer_predictions,
                        depths_materials_columns_pairs=page_predictions["depths_materials_column_pairs"],
                    )
                )
            else:
                page_predictions_class.append(PagePredictions(page_number=page_number, layers=layer_predictions))

        return FilePredictions(pages=page_predictions_class, file_name=file_name, language=file_language)

    def evaluate(self, ground_truth_layers: list):
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
