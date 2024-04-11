"""This module contains classes for predictions."""

import fitz
import Levenshtein

from stratigraphy.benchmark.ground_truth import GroundTruthForFile
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.line import TextLine, TextWord
from stratigraphy.util.util import parse_text


class MaterialDescriptionPrediction:
    """A class to represent a material description prediction.

    TODO: Check if this class can be replaced.
    """

    def __init__(self, text: str, rect: list, lines: list):
        self.text = text
        self.rect = fitz.Rect(rect)
        self.lines = [TextLine([TextWord(**line)]) for line in lines]


class DepthIntervalPrediction:
    """A class to represent a depth interval prediction.

    TODO: Could be replaced by BoundaryInterval from interval.py. Then we could use line_anchor and background_rect.
    """

    def __init__(self, start: dict, end: dict):
        self.start = (
            DepthColumnEntry(value=start["value"], rect=fitz.Rect(start["rect"])) if start is not None else None
        )
        self.end = DepthColumnEntry(value=end["value"], rect=fitz.Rect(end["rect"])) if end is not None else None


class LayerPrediction:
    """A class to represent predictions for a single layer."""

    def __init__(self, material_description: MaterialDescriptionPrediction, depth_interval: DepthIntervalPrediction):
        self.material_description = material_description
        self.depth_interval = depth_interval
        self.material_is_correct = None
        self.depth_interval_is_correct = None


class PagePredictions:
    """A class to represent predictions for a single page."""

    def __init__(
        self, layers: list[LayerPrediction], page_number: int, depths_materials_columns_pairs: list[dict] = None
    ):
        self.layers = layers
        self.page_number = page_number
        self.depths_materials_columns_pairs = depths_materials_columns_pairs


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(self, pages: list[PagePredictions] = None, file_name: str = None):
        self.pages = pages
        self.file_name = file_name
        if self.pages:
            self.layers = sum([page.layers for page in self.pages], [])

    def create_from_json(self, predictions_for_file: dict, file_name: str):
        """Create predictions class for a file given the predictions.json file.

        Args:
            predictions_for_file (dict): The predictions for the file in json format.
            file_name (str): The name of the file.
        """
        page_predictions_class = []
        for page_number, page_predictions in predictions_for_file.items():
            page_layers = page_predictions["layers"]
            layer_predictions = []
            for layer in page_layers:
                material_prediction = MaterialDescriptionPrediction(**layer["material_description"])
                if "depth_interval" in layer:
                    depth_interval_prediction = DepthIntervalPrediction(**layer["depth_interval"])
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

        self.file_name = file_name
        self.pages = page_predictions_class
        self.layers = sum([page.layers for page in self.pages], [])

    def evaluate(self, ground_truth: GroundTruthForFile):
        """Evaluate all layers of the predictions against the ground truth.

        Args:
            ground_truth (GroundTruthForFile): The ground truth for the file.
        """
        self._check_if_initialized()
        self.unmatched_layers = ground_truth.layers.copy()
        for layer in self.layers:
            match, depth_interval_is_correct = self._find_matching_layer(layer)
            if match:
                layer.material_is_correct = True
                layer.depth_interval_is_correct = depth_interval_is_correct
            else:
                layer.material_is_correct = False
                layer.depth_interval_is_correct = None

    def _find_matching_layer(self, layer: LayerPrediction) -> tuple[dict, bool]:
        """Find the matching layer in the ground truth.

        Args:
            layer (LayerPrediction): The layer to match.

        Returns:
            tuple[dict, bool]: The matching layer and a boolean indicating if the depth interval is correct.
        """
        self._check_if_initialized()
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
                start == 0 and layer.depth_interval.start is None and end == layer.depth_interval.end
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

    def _check_if_initialized(self):
        if self.pages is None:
            raise ValueError(
                "No predictions found for this file. Initialize the predictions first."
                "You may use the create_from_json() method."
            )
