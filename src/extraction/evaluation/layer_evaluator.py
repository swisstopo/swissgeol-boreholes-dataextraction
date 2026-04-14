"""Classes for evaluating the groundwater levels of a borehole."""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import Levenshtein

from core.benchmark_utils import Metrics
from extraction.features.predictions.borehole_predictions import (
    BoreholePredictionsWithGroundTruth,
    FileLayersWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.stratigraphy.layer.layer import Layer
from extraction.utils.dynamic_matching import PredToGroundTruthLayerDP
from swissgeol_doc_processing.utils.file_utils import parse_text

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9
MAX_DEPTH_SCORE = 1.0


class LayerEvaluator:
    """Class for evaluating the layer information of all boreholes in a document."""

    @staticmethod
    def get_layer_metrics(layers: FileLayersWithGroundTruth) -> Metrics:
        """Calculate layer-level metrics for the given file's predictions.

        Args:
            layers (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: The computed layer metrics.
        """
        return LayerEvaluator.calculate_metrics(
            layers=layers,
            num_ground_truth_fn=lambda ground_truth_layers: len(ground_truth_layers),
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.is_correct,
        )

    @staticmethod
    def get_material_description_metrics(layers: FileLayersWithGroundTruth) -> Metrics:
        """Calculate metrics for material description extraction across all boreholes in a file.

        Args:
            layers (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: Aggregated material metrics across all boreholes.
        """

        def per_layer_action(layer: Layer):
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")

        def num_ground_truth_fn(ground_truth_layers: list[dict]):
            return sum(lay["material_description"] is not None for lay in ground_truth_layers)

        return LayerEvaluator.calculate_metrics(
            layers=layers,
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.material_description.is_correct,
            per_layer_action=per_layer_action,
        )

    @staticmethod
    def get_depth_interval_metrics(layers: FileLayersWithGroundTruth) -> Metrics:
        """Calculate metrics for depth interval extraction across all boreholes in a file.

        Args:
            layers (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: Aggregated depth metrics across all boreholes.
        """

        def num_ground_truth_fn(ground_truth_layers: list[dict]):
            return sum(lay["depth_interval"] is not None for lay in ground_truth_layers)

        return LayerEvaluator.calculate_metrics(
            layers=layers,
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.depths is not None and layer.depths.is_correct,
        )

    @staticmethod
    def calculate_metrics(
        layers: FileLayersWithGroundTruth,
        num_ground_truth_fn: Callable[[list[dict]], int],
        per_layer_filter: Callable[[Layer], bool],
        per_layer_condition: Callable[[Layer], bool],
        per_layer_action: Callable[[Layer], None] | None = None,
    ) -> Metrics:
        """Calculate metrics based on a condition per layer, after applying a filter.

        Args:
            layers (FileLayersWithGroundTruth): Borehole layer predictions paired with ground truth.
            num_ground_truth_fn (Callable[[list[dict]], int]): Function that returns the number of ground truth.
            per_layer_filter (Callable[[Layer], bool]): Function to filter layers to consider.
            per_layer_condition (Callable[[Layer], bool]): Function that returns True if the layer is a hit.
            per_layer_action (Optional[Callable[[Layer], None]]): Optional action to perform per layer.

        Returns:
            Metrics: The calculated metrics.
        """
        hits_for_all_borehole = 0
        total_predictions_for_all_boreholes = 0
        fn_for_all_boreholes = 0

        for borehole_data in layers.boreholes:
            number_of_truth_values = num_ground_truth_fn(borehole_data.ground_truth)
            tp = 0
            total_predictions = 0

            layers = borehole_data.layers.layers if borehole_data.layers else []
            for layer in layers:
                if per_layer_action:
                    per_layer_action(layer)
                if per_layer_filter(layer):
                    total_predictions += 1
                    if per_layer_condition(layer):
                        tp += 1

            fn = number_of_truth_values - tp

            hits_for_all_borehole += tp
            total_predictions_for_all_boreholes += total_predictions
            fn_for_all_boreholes += fn

        # at this point we have the global statistics for all the boreholes in the document
        return Metrics(
            tp=hits_for_all_borehole,
            fp=total_predictions_for_all_boreholes - hits_for_all_borehole,
            fn=fn_for_all_boreholes,
        )

    @staticmethod
    def evaluate(file_predictions: FileLayersWithGroundTruth) -> tuple[Metrics, Metrics, Metrics]:
        """Evaluate all predicted layers for a borehole against the ground truth.

        Args:
            file_predictions (FileLayersWithGroundTruth): Layer predictions with ground truth,
                grouped by borehole.

        Returns:
            tuple[Metrics, Metrics, Metrics]: (layer_metrics, depth_interval_metrics, material_description_metrics)
        """

        # Utility functions to set correctness flags on predicted layers
        def set_depths_flag(predicted_layer, ground_truth_layer):
            if predicted_layer.depths is not None:
                predicted_layer.depths.is_correct = (
                    score_depths(predicted_layer, ground_truth_layer) == MAX_DEPTH_SCORE
                )

        def set_material_description_flag(predicted_layer, groud_truth_layers):
            predicted_layer.material_description.is_correct = (
                score_material_descriptions(predicted_layer, groud_truth_layers)
                >= MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
            )

        def set_layer_flag(predicted_layer, groud_truth_layers):
            predicted_layer.is_correct = (
                score_depths(predicted_layer, groud_truth_layers) == MAX_DEPTH_SCORE
                and score_material_descriptions(predicted_layer, groud_truth_layers)
                >= MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
            )

        for borehole_data in file_predictions.boreholes:
            if borehole_data.layers:
                predicted_layers = borehole_data.layers.layers

                for pred in predicted_layers:
                    pred.material_description.is_correct = False
                    if pred.depths is not None:
                        pred.depths.is_correct = False
                    pred.is_correct = False

                LayerEvaluator.apply_mapping(
                    borehole_data.ground_truth, predicted_layers, score_depths, set_depths_flag
                )
                LayerEvaluator.apply_mapping(
                    borehole_data.ground_truth,
                    predicted_layers,
                    score_material_descriptions,
                    set_material_description_flag,
                )
                LayerEvaluator.apply_mapping(borehole_data.ground_truth, predicted_layers, score_layer, set_layer_flag)

        layer_metrics = LayerEvaluator.get_layer_metrics(file_predictions)
        depth_interval_metrics = LayerEvaluator.get_depth_interval_metrics(file_predictions)
        material_description_metrics = LayerEvaluator.get_material_description_metrics(file_predictions)

        return layer_metrics, depth_interval_metrics, material_description_metrics

    @staticmethod
    def apply_mapping(ground_truth_layers, predicted_layers, scoring_fn, set_flag_fn):
        _, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(
            ground_truth_layers, predicted_layers, scoring_fn
        )
        for predicted_layer, ground_truth_layer in mapping:
            set_flag_fn(predicted_layer, ground_truth_layer)

    @staticmethod
    def match_boreholes_to_ground_truth(
        file_predictions: FilePredictions, ground_truth_for_file: dict
    ) -> list[BoreholePredictionsWithGroundTruth]:
        """Match predicted boreholes to ground truth boreholes.

        This method compares the predicted boreholes with the ground truth boreholes  and establishes a mapping
            between them based on their similarity.

        Args:
            file_predictions (FilePredictions): all predictions for the file
            ground_truth_for_file (dict): the ground truth for the file

        Returns:
            list[BoreholePredictionsWithGroundTruth] : A list of matched borehole predictions with their ground truth.
        """
        all_ground_truth_layers = {
            idx: borehole_data["layers"] for idx, borehole_data in ground_truth_for_file.items()
        }
        borehole_layers = [bh.layers_in_borehole for bh in file_predictions.borehole_predictions_list]
        pred_vs_gt_matching_score = defaultdict(dict)
        for gt_idx, ground_truth_layers in all_ground_truth_layers.items():
            for pred_idx, predicted_layers in enumerate(borehole_layers):
                matching_score, _ = LayerEvaluator.compute_borehole_affinity_and_mapping(
                    ground_truth_layers, predicted_layers.layers, score_layer
                )
                pred_vs_gt_matching_score[gt_idx][pred_idx] = matching_score

        # matching of all the boreholes detected to a borehole in the ground truth
        matched_boreholes = []
        assigned_preds = set()
        while pred_vs_gt_matching_score:
            max_score = float("-inf")
            for gt_idx, pred_scores in pred_vs_gt_matching_score.items():
                for pred_idx, score in pred_scores.items():
                    if score > max_score and pred_idx not in assigned_preds:  # can't assign the same pred twice
                        max_score = score
                        best_matches = (gt_idx, pred_idx)

            gt_best_idx, pred_best_idx = best_matches
            matched_boreholes.append(
                BoreholePredictionsWithGroundTruth(
                    file_predictions.borehole_predictions_list[pred_best_idx], ground_truth_for_file[gt_best_idx]
                )
            )
            assigned_preds.add(pred_best_idx)  # Mark this pred_idx as used

            # Remove the matched gt_idx from consideration
            del pred_vs_gt_matching_score[gt_best_idx]

            if len(assigned_preds) == len(borehole_layers):
                # all preds have been assigned
                break

        # add entries with missing predictions for all unmatched ground truth boreholes (will count as false negatives)
        for gt_idx in pred_vs_gt_matching_score:
            matched_boreholes.append(
                BoreholePredictionsWithGroundTruth(predictions=None, ground_truth=ground_truth_for_file[gt_idx])
            )

        # add entries with missing ground truth for all unmatched prediction boreholes (will count as false positives)
        for index, pred in enumerate(file_predictions.borehole_predictions_list):
            if index not in assigned_preds:
                matched_boreholes.append(BoreholePredictionsWithGroundTruth(predictions=pred, ground_truth={}))
        return matched_boreholes

    @staticmethod
    def compute_borehole_affinity_and_mapping(
        ground_truth_layers: list[dict[str, Any]],
        predicted_layers: list[Layer],
        scoring_fn: Callable[[Layer, dict], float],
    ) -> tuple[float, list[tuple[Layer, dict]]]:
        """Computes the matching score between a prediction and a groundtruth borehole.

        Computing this score allows to match the predictions identified in the document against the correct
        groundtruth. The matching score is computed by comparing the layers of each borehole identified to each
        layers in the ground truth.

        Args:
            ground_truth_layers (list[dict]): list containing the ground truth for the layers
            predicted_layers (list[Layer]): object containing the list of the predicted layers
            scoring_fn: Callable[[Layer, dict], float]: the scoring function to be used for selecting the best mapping

        Returns:
            tuple: containing
                - matching_score (float): a score that captures the similarity between the predicted and ground
                    truth layers. Maximum is 1.0.
                - mapping (list[(Layer, dict)]): a list of mappings between predicted and ground truth layers.
        """
        dp = PredToGroundTruthLayerDP(predicted_layers, ground_truth_layers, [0.0] * len(ground_truth_layers))
        return dp.solve(scoring_fn)


def score_material_descriptions(layer: Layer, ground_truth: dict) -> float:
    """Scores how well the extracted material description matches the ground truth on a scale from 0 to 1."""
    parsed_text = parse_text(layer.material_description.text)
    return Levenshtein.ratio(parsed_text, parse_text(ground_truth["material_description"]))


def score_depths(layer: Layer, ground_truth: dict) -> float:
    """Scores how well the extracted depths match the ground truth on a scale from 0 to 1.

    The total score is composed of 0.5 for matching start and 0.5 for matching end.
    """
    depth_score = 0.0
    ground_truth_start = ground_truth["depth_interval"]["start"]
    ground_truth_end = ground_truth["depth_interval"]["end"]

    if layer.depths is not None:
        if (layer.depths.start is None and ground_truth_start is None) or (
            layer.depths.start is not None and layer.depths.start.value == ground_truth_start
        ):
            depth_score += 0.5

        if (layer.depths.end is None and ground_truth_end is None) or (
            layer.depths.end is not None and layer.depths.end.value == ground_truth_end
        ):
            depth_score += 0.5
    else:
        if ground_truth_start is None and ground_truth_end is None:
            depth_score += 1
    return depth_score


def score_layer(layer: Layer, ground_truth: dict) -> float:
    """Scores how well the full layer matches the ground truth on a scale from 0 to 1."""
    return (score_material_descriptions(layer, ground_truth) + score_depths(layer, ground_truth)) / 2
