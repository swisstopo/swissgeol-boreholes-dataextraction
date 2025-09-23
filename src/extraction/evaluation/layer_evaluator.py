"""Classes for evaluating the groundwater levels of a borehole."""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import Levenshtein

from extraction.evaluation.benchmark.metrics import OverallMetrics
from extraction.evaluation.evaluation_dataclasses import Metrics
from extraction.features.predictions.borehole_predictions import (
    BoreholePredictionsWithGroundTruth,
    FileLayersWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.stratigraphy.layer.layer import Layer
from utils.file_utils import parse_text

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9
MAX_DEPTH_SCORE = 1.0


class LayerEvaluator:
    """Class for evaluating the layer information of all boreholes in a document."""

    def __init__(
        self,
        file_layers_list: list[FileLayersWithGroundTruth],
    ):
        """Initializes the LayerEvaluator object.

        Args:
            file_layers_list (list[FileLayersWithGroundTruth]): The layers to evaluate, grouped by borehole in a list,
                with associated ground truth data for each borehole.
        """
        self.file_layers_list = file_layers_list

    def get_layer_metrics(self) -> OverallMetrics:
        return self.calculate_metrics(
            num_ground_truth_fn=lambda ground_truth_layers: len(ground_truth_layers),
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.is_correct,
        )

    def get_material_description_metrics(self) -> OverallMetrics:
        """Calculate metrics for layer predictions."""

        def per_layer_action(layer: Layer):
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")

        def num_ground_truth_fn(ground_truth_layers: list[dict]):
            return sum(lay["material_description"] is not None for lay in ground_truth_layers)

        return self.calculate_metrics(
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.material_description.is_correct,
            per_layer_action=per_layer_action,
        )

    def get_depth_interval_metrics(self) -> OverallMetrics:
        """Calculate metrics for depth interval predictions."""

        def num_ground_truth_fn(ground_truth_layers: list[dict]):
            return sum(lay["depth_interval"] is not None for lay in ground_truth_layers)

        return self.calculate_metrics(
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.depths is not None and layer.depths.is_correct,
        )

    def calculate_metrics(
        self,
        num_ground_truth_fn: Callable[[list[dict]], int],
        per_layer_filter: Callable[[Layer], bool],
        per_layer_condition: Callable[[Layer], bool],
        per_layer_action: Callable[[Layer], None] | None = None,
    ) -> OverallMetrics:
        """Calculate metrics based on a condition per layer, after applying a filter.

        Args:
            num_ground_truth_fn (Callable[[list[dict]], int]): Function that returns the number of ground truth.
            per_layer_filter (Callable[[LayerPrediction], bool]): Function to filter layers to consider.
            per_layer_condition (Callable[[LayerPrediction], bool]): Function that returns True if the layer is a hit.
            per_layer_action (Optional[Callable[[LayerPrediction], None]]): Optional action to perform per layer.

        Returns:
            OverallMetrics: The calculated metrics.
        """
        overall_metrics = OverallMetrics()

        # iteration over all the files
        for file in self.file_layers_list:
            hits_for_all_borehole = 0
            total_predictions_for_all_boreholes = 0
            fn_for_all_boreholes = 0

            for borehole_data in file.boreholes:
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
            overall_metrics.metrics[file.filename] = Metrics(
                tp=hits_for_all_borehole,
                fp=total_predictions_for_all_boreholes - hits_for_all_borehole,
                fn=fn_for_all_boreholes,
            )

        return overall_metrics

    @staticmethod
    def match_predictions_with_ground_truth(file_predictions: FilePredictions, ground_truth_for_file: dict):
        """Evaluate all predicted layers for a borehole against the ground truth.

        Also performs the matching groundtruth to prediction when there is more than one borehole in the document.
        It is for this reason that the layers are the first element that needs to be elaluated.

        Args:
            file_predictions (FilePredictions): all predictions for the file
            ground_truth_for_file (dict): the ground truth for the file

        Returns:
            list[BoreholePredictionsWithGroundTruth]
        """
        matched_boreholes = LayerEvaluator.match_boreholes_to_ground_truth(file_predictions, ground_truth_for_file)

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

        # now compute the real statistics for the matched pairs of boreholes
        for borehole_data in matched_boreholes:
            if borehole_data.predictions:
                predicted_layers = borehole_data.predictions.layers_in_borehole.layers

                for pred in predicted_layers:
                    pred.material_description.is_correct = False
                    if pred.depths is not None:
                        pred.depths.is_correct = False
                    pred.is_correct = False

                ground_truth_layers = borehole_data.ground_truth.get("layers", [])

                LayerEvaluator.apply_mapping(ground_truth_layers, predicted_layers, score_depths, set_depths_flag)
                LayerEvaluator.apply_mapping(
                    ground_truth_layers, predicted_layers, score_material_descriptions, set_material_description_flag
                )
                LayerEvaluator.apply_mapping(ground_truth_layers, predicted_layers, score_layer, set_layer_flag)

        return matched_boreholes

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
    ) -> tuple[float, list[(Layer, dict)]]:
        """Computes the matching score between a prediction and a groundtruth borehole.

        This is relevant when there is more than one borehole per pdf. Computing this score allows to match the
        predictions identified in the document against the correct groundtruth. The matching score is computed by
        comparing the layers of each borehole identified to each layers in the groudtruth.

        Args:
            ground_truth_layers (list[dict]): list containing the ground truth for the layers
            predicted_layers (LayersInBorehole): object containing the list of the predicted layers
            scoring_fn: Callable[[Layer, dict], float]: the scoring function to be used for selecting the best mapping

        Returns:
            tuple: containing
                - matching_score (float): a score that captures the similarity between the predicted and ground
                    truth layers. Maximum is 1.0.
                - mapping (list[(Layer, dict)]): a list of mappings between predicted and ground truth layers.
        """
        if not predicted_layers or not ground_truth_layers:
            return 0.0, []

        P, G = len(predicted_layers), len(ground_truth_layers)

        # Precompute pairwise scores
        pairwise_scores = LayerEvaluator._compute_scores(ground_truth_layers, predicted_layers, P, G, scoring_fn)

        # Build DP table and pointer
        dp, ptr = LayerEvaluator._build_dp_table(P, G, pairwise_scores)

        # Backtrack to recover mapping
        mapping = LayerEvaluator._get_mapping(P, G, ptr)
        layer_mapping = [
            (predicted_layers[predicted_layer_index], ground_truth_layers[ground_truth_layer_index])
            for predicted_layer_index, ground_truth_layer_index in mapping
        ]

        matching_score = dp[P][G] / max(P, G)  # maximum is 1.
        return matching_score, layer_mapping

    @staticmethod
    def _compute_scores(
        ground_truth_layers: list[dict[str, Any]],
        preds: list[Layer],
        P: int,
        G: int,
        scoring_fn: Callable[[Layer, dict], float],
    ) -> list[list[float]]:
        """Compute pairwise scores between predicted and ground truth layers.

        Args:
            ground_truth_layers (list[dict[str, Any]]): The ground truth layers with
                material_description and depth_interval fields.
            preds (list[Layer]): The predicted layers to evaluate.
            P (int): The number of predicted layers.
            G (int): The number of ground truth layers.
            scoring_fn: Callable[[Layer, dict], float]: the scoring function to be used for selecting the best mapping.

        Returns:
            list[list[float]]: scores for each pred-gt pair.
        """
        pair_score = [[0.0] * G for _ in range(P)]

        for i in range(P):
            for j in range(G):
                pair_score[i][j] = scoring_fn(preds[i], ground_truth_layers[j])
        return pair_score

    @staticmethod
    def _build_dp_table(P: int, G: int, pair_score: list[list[float]]) -> tuple[list[list[float]], list[list[str]]]:
        """Build the dynamic programming table for layer matching.

        The algorithm finds the optimal alignment between predicted and ground truth layers
        while preserving their relative order. It uses a dynamic programming approach where:
        - Each cell dp[i][j] represents the best cumulative score for matching the first i
          predictions with the first j ground truth layers
        - Moves can be:
          * Diagonal: Match current prediction with current ground truth (score from pair_score)
          * Up: Skip current prediction (no additional score)
          * Left: Skip current ground truth (no additional score)
        - In case of equal scores, diagonal moves are preferred to preserve matching,
          followed by up moves, then left moves.

        Args:
            P (int): The number of predicted layers.
            G (int): The number of ground truth layers.
            pair_score (list[list[float]]): The pairwise scores between predicted and ground truth layers.
                Each score combines text similarity and depth matching accuracy.

        Returns:
            tuple: A tuple containing:
                - dp (list[list[float]]): The dynamic programming table with cumulative scores
                - ptr (list[list[str]]): The pointer table storing move directions ('diag', 'up', 'left')
                    used for backtracking to recover the optimal matching.
        """
        dp = [[0.0] * (G + 1) for _ in range(P + 1)]
        ptr = [["None"] * (G + 1) for _ in range(P + 1)]

        for i in range(1, P + 1):
            for j in range(1, G + 1):
                diag = dp[i - 1][j - 1] + pair_score[i - 1][j - 1]
                up = dp[i - 1][j]
                left = dp[i][j - 1]
                # tie-break: diag > up > left
                candidates = [("diag", diag, 3), ("up", up, 2), ("left", left, 1)]
                choice = max(candidates, key=lambda x: (x[1], x[2]))  # break tie with prioritize matching
                dp[i][j] = choice[1]
                ptr[i][j] = choice[0]
        return dp, ptr

    @staticmethod
    def _get_mapping(
        P: int,
        G: int,
        ptr: list[list[str]],
    ) -> list[(int, int)]:
        """Get the mapping between predicted and ground truth layers by backtracking through the DP table.

        Args:
            P (int): The number of predicted layers.
            G (int): The number of ground truth layers.
            ptr (list[list[str]]): The pointer table for backtracking.

        Returns:
            list[(int, int)]: The mapping between predicted and ground truth layers.
        """
        i, j = P, G
        mapping = []
        while i > 0 and j > 0:
            move = ptr[i][j]
            if move == "diag":
                predicted_layer_index, ground_truth_layer_index = i - 1, j - 1
                mapping.append((predicted_layer_index, ground_truth_layer_index))
                i, j = i - 1, j - 1
            elif move == "up":
                i -= 1
            elif move == "left":
                j -= 1
            else:
                break
        mapping.reverse()
        return mapping


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
        if (layer.depths.start is None and (ground_truth_start == 0 or ground_truth_start is None)) or (
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
