"""Classes for evaluating the groundwater levels of a borehole."""

import logging
from collections.abc import Callable

import Levenshtein
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetrics
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.utility import _is_valid_depth_interval
from stratigraphy.layer.layer import Layer, LayersInDocument
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9


class LayerEvaluator:
    """Class for evaluating the extracted groundwater information of a borehole."""

    def __init__(self, layers_entries: list[LayersInDocument], ground_truth: GroundTruth):
        """Initializes the LayerEvaluator object.

        Args:
            layers_entries (list[LayersInDocument]): The layers to evaluate.
            ground_truth (GroundTruth): The ground truth.
        """
        self.ground_truth = ground_truth
        self.layers_entries: list[LayersInDocument] = layers_entries

    def get_layer_metrics(self) -> OverallMetrics:
        """Calculate metrics for layer predictions."""

        def per_layer_action(layer):
            if parse_text(layer.material_description.feature.text) == "":
                logger.warning("Empty string found in predictions")

        return self.calculate_metrics(
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.material_description.feature.is_correct,
            per_layer_action=per_layer_action,
        )

    def get_depth_interval_metrics(self) -> OverallMetrics:
        """Calculate metrics for depth interval predictions."""
        return self.calculate_metrics(
            per_layer_filter=lambda layer: layer.material_description.feature.is_correct
            and layer.is_correct is not None,
            per_layer_condition=lambda layer: layer.is_correct,
        )

    def calculate_metrics(
        self,
        per_layer_filter: Callable[[Layer], bool],
        per_layer_condition: Callable[[Layer], bool],
        per_layer_action: Callable[[Layer], None] | None = None,
    ) -> OverallMetrics:
        """Calculate metrics based on a condition per layer, after applying a filter.

        Args:
            per_layer_filter (Callable[[LayerPrediction], bool]): Function to filter layers to consider.
            per_layer_condition (Callable[[LayerPrediction], bool]): Function that returns True if the layer is a hit.
            per_layer_action (Optional[Callable[[LayerPrediction], None]]): Optional action to perform per layer.

        Returns:
            OverallMetrics: The calculated metrics.
        """
        overall_metrics = OverallMetrics()

        for layers_in_document in self.layers_entries:
            ground_truth_for_file = self.ground_truth.for_file(layers_in_document.filename)
            number_of_truth_values = len(ground_truth_for_file["layers"])
            hits = 0
            total_predictions = 0

            for layer in layers_in_document.layers:
                if per_layer_action:
                    per_layer_action(layer)
                if per_layer_filter(layer):
                    total_predictions += 1
                    if per_layer_condition(layer):
                        hits += 1

            fn = 0
            fn = number_of_truth_values - hits

            if total_predictions > 0:
                overall_metrics.metrics[layers_in_document.filename] = Metrics(
                    tp=hits,
                    fp=total_predictions - hits,
                    fn=fn,
                )

        return overall_metrics

    @staticmethod
    def evaluate_borehole(predicted_layers: list[Layer], ground_truth_layers: list):
        """Evaluate all predicted layers for a borehole against the ground truth.

        Args:
            predicted_layers (list[Layer]): The predicted layers for the borehole.
            ground_truth_layers (list): The ground truth layers for the borehole.
        """
        unmatched_layers = ground_truth_layers.copy()
        for layer in predicted_layers:
            match, depth_interval_is_correct = LayerEvaluator.find_matching_layer(layer, unmatched_layers)
            if match:
                layer.material_description.feature.is_correct = True
                layer.is_correct = depth_interval_is_correct
            else:
                layer.material_description.feature.is_correct = False
                layer.is_correct = None

    @staticmethod
    def find_matching_layer(layer: Layer, unmatched_layers: list[dict]) -> tuple[dict, bool] | tuple[None, None]:
        """Find the matching layer in the ground truth, if any, and remove it from the list of unmatched layers.

        Args:
            layer (Layer): The layer to match.
            unmatched_layers (list[dict]): The layers from the ground truth that were not yet matched during the
                                            current evaluation.

        Returns:
            tuple[dict, bool] | tuple[None, None]: The matching layer and a boolean indicating if the depth interval
                                is correct. None if no match was found.
        """
        parsed_text = parse_text(layer.material_description.feature.text)
        possible_matches = [
            ground_truth_layer
            for ground_truth_layer in unmatched_layers
            if Levenshtein.ratio(parsed_text, ground_truth_layer["material_description"])
            > MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
        ]

        if not possible_matches:
            return None, None

        for possible_match in possible_matches:
            start = possible_match["depth_interval"]["start"]
            end = possible_match["depth_interval"]["end"]

            if _is_valid_depth_interval(layer.depths, start, end):
                unmatched_layers.remove(possible_match)
                return possible_match, True

        match = max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, x["material_description"]))
        unmatched_layers.remove(match)
        return match, False
