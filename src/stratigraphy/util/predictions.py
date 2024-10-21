"""This module contains classes for predictions."""

import logging
from collections import Counter
from pathlib import Path

import Levenshtein

from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPairs
from stratigraphy.evaluation.evaluation_dataclasses import Metrics, OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInformation, GroundwaterInformationOnPage
from stratigraphy.layer.layer import LayerPrediction
from stratigraphy.metadata.metadata import BoreholeMetadata, BoreholeMetadataList
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers: list[LayerPrediction] | None,
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater_entries: list[GroundwaterInformationOnPage] | None,
        depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs] | None,
    ):
        self.layers: list[LayerPrediction] | None = layers
        self.depths_materials_columns_pairs: list[DepthsMaterialsColumnPairs] | None = depths_materials_columns_pairs
        self.file_name: str = file_name
        self.metadata: BoreholeMetadata = metadata
        self.groundwater_entries: list[GroundwaterInformationOnPage] | None = groundwater_entries
        self.groundwater_is_correct: dict = {}

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
        if self.metadata is not None and self.metadata.coordinates is not None:
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
        groundwater_ground_truth = ground_truth.get("groundwater", [])
        if groundwater_ground_truth is None:
            groundwater_ground_truth = []
        self.evaluate_groundwater(groundwater_ground_truth)

    def evaluate_layers(self, ground_truth_layers: list):
        """Evaluate all layers of the predictions against the ground truth.

        Args:
            ground_truth_layers (list): The ground truth layers for the file.
        """
        unmatched_layers = ground_truth_layers.copy()
        for layer in self.layers:
            match, depth_interval_is_correct = self._find_matching_layer(layer, unmatched_layers)
            if match:
                layer.material_is_correct = True
                layer.depth_interval_is_correct = depth_interval_is_correct
            else:
                layer.material_is_correct = False
                layer.depth_interval_is_correct = None

    @staticmethod
    def count_against_ground_truth(values: list, ground_truth: list) -> Metrics:
        """Count the number of true positives, false positives and false negatives.

        Args:
            values (list): The values to count.
            ground_truth (list): The ground truth values.

        Returns:
            Metrics: The metrics for the values.
        """
        # Counter deals with duplicates when doing intersection
        values_counter = Counter(values)
        ground_truth_counter = Counter(ground_truth)

        tp = (values_counter & ground_truth_counter).total()  # size of intersection
        return Metrics(tp=tp, fp=len(values) - tp, fn=len(ground_truth) - tp)

    def evaluate_groundwater(self, groundwater_ground_truth: list):
        """Evaluate the groundwater information of the file against the ground truth.

        Args:
            groundwater_ground_truth (list): The ground truth for the file.
        """
        ############################################################################################################
        ### Compute the metadata correctness for the groundwater information.
        ############################################################################################################
        gt_groundwater = [
            GroundwaterInformation.from_json_values(
                depth=json_gt_data["depth"],
                date=json_gt_data["date"],
                elevation=json_gt_data["elevation"],
            )
            for json_gt_data in groundwater_ground_truth
        ]

        self.groundwater_is_correct["groundwater"] = self.count_against_ground_truth(
            [
                (
                    entry.groundwater.depth,
                    entry.groundwater.format_date(),
                    entry.groundwater.elevation,
                )
                for entry in self.groundwater_entries
            ],
            [(entry.depth, entry.format_date(), entry.elevation) for entry in gt_groundwater],
        )
        self.groundwater_is_correct["groundwater_depth"] = self.count_against_ground_truth(
            [entry.groundwater.depth for entry in self.groundwater_entries],
            [entry.depth for entry in gt_groundwater],
        )
        self.groundwater_is_correct["groundwater_elevation"] = self.count_against_ground_truth(
            [entry.groundwater.elevation for entry in self.groundwater_entries],
            [entry.elevation for entry in gt_groundwater],
        )
        self.groundwater_is_correct["groundwater_date"] = self.count_against_ground_truth(
            [entry.groundwater.date for entry in self.groundwater_entries],
            [entry.date for entry in gt_groundwater],
        )

    @staticmethod
    def _find_matching_layer(
        layer: LayerPrediction, unmatched_layers: list[dict]
    ) -> tuple[dict, bool] | tuple[None, None]:
        """Find the matching layer in the ground truth.

        Args:
            layer (LayerPrediction): The layer to match.
            unmatched_layers (list[dict]): The layers from the ground truth that were not yet matched during the
                                           current evaluation.

        Returns:
            tuple[dict, bool] | tuple[None, None]: The matching layer and a boolean indicating if the depth interval
                                is correct. None if no match was found.
        """
        parsed_text = parse_text(layer.material_description.text)
        possible_matches = [
            ground_truth_layer
            for ground_truth_layer in unmatched_layers
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
                unmatched_layers.remove(possible_match)
                return possible_match, True

            elif (  # noqa: SIM102
                layer.depth_interval.start is not None and layer.depth_interval.end is not None
            ):  # In all other cases we do not allow a None value.
                if start == layer.depth_interval.start.value and end == layer.depth_interval.end.value:
                    unmatched_layers.remove(possible_match)
                    return possible_match, True

        match = max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, x["material_description"]))
        unmatched_layers.remove(match)
        return match, False

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            self.file_name: {
                "metadata": self.metadata.to_json(),
                "layers": [layer.to_json() for layer in self.layers] if self.layers is not None else [],
                "depths_materials_column_pairs": [
                    dmc_pair.to_json() for dmc_pair in self.depths_materials_columns_pairs
                ]
                if self.depths_materials_columns_pairs is not None
                else [],
                "page_dimensions": self.metadata.page_dimensions,  # TODO: Remove, already in metadata
                "groundwater": [entry.to_json() for entry in self.groundwater_entries]
                if self.groundwater_entries is not None
                else [],
                "file_name": self.file_name,
            }
        }


class OverallFilePredictions:
    """A class to represent predictions for all files."""

    def __init__(self):
        """Initializes the OverallFilePredictions object."""
        self.file_predictions_list: list[FilePredictions] = []

    def add_file_predictions(self, file_predictions: FilePredictions):
        """Add file predictions to the list of file predictions.

        Args:
            file_predictions (FilePredictions): The file predictions to add.
        """
        self.file_predictions_list.append(file_predictions)

    def get_metadata_as_dict(self):
        """Returns the metadata of the predictions as a dictionary."""
        return {
            file_prediction.file_name: file_prediction.metadata.to_json()
            for file_prediction in self.file_predictions_list
        }

    def to_json(self) -> dict:
        """Converts the object to a dictionary by merging individual file predictions.

        Returns:
            dict: A dictionary representation of the object.
        """
        return {k: v for fp in self.file_predictions_list for k, v in fp.to_json().items()}

    def evaluate_metadata_extraction(self, ground_truth_path: Path) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            ground_truth_path (Path): The path to the ground truth file.
        """
        metadata_per_file: BoreholeMetadataList = BoreholeMetadataList()

        for file_prediction in self.file_predictions_list:
            metadata_per_file.add_metadata(file_prediction.metadata)
        return MetadataEvaluator(metadata_per_file, ground_truth_path).evaluate()
