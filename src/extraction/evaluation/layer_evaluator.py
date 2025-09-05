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
                    else:
                        raise ValueError("NO GT")

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

        # now compute the real statistics for the matched pairs of boreholes
        for borehole_data in matched_boreholes:
            if borehole_data.predictions:
                predicted_layers = borehole_data.predictions.layers_in_borehole.layers
                groud_truth_layers = borehole_data.ground_truth.get("layers", [])
                # _, mapping = LayerEvaluator.compute_borehole_affinity(groud_truth_layers, predicted_layers)  # old
                _, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(groud_truth_layers, predicted_layers)
                # This method modifies the internal state of predictions and updates their flags
                LayerEvaluator._update_prediction_flags(predicted_layers, mapping)
        return matched_boreholes

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
                # matching_score, _ = LayerEvaluator.compute_borehole_affinity( # OLD
                #     ground_truth_layers, deepcopy(predicted_layers.layers)
                # )
                matching_score, _ = LayerEvaluator.compute_borehole_affinity_and_mapping(
                    ground_truth_layers, predicted_layers.layers
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
    ) -> tuple[float, list[dict]]:
        """Computes the matching score between a prediction and a groundtruth borehole.

        This is relevant when there is more than one borehole per pdf. Computing this score allows to match the
        predictions identified in the document against the correct groundtruth. The matching score is computed by
        comparing the layers of each borehole identified to each layers in the groudtruth.

        Args:
            ground_truth_layers (list[dict]): list containing the ground truth for the layers
            predicted_layers (LayersInBorehole): object containing the list of the predicted layers

        Returns:
            tuple: containing
                - matching_score (float): a score that captures the similarity between the predicted and ground
                    truth layers. Maximum is 1.0.
                - mapping (list[dict]): a list of mappings between predicted and ground truth layers.
        """
        if not predicted_layers or not ground_truth_layers:
            return 0.0, []

        P, G = len(predicted_layers), len(ground_truth_layers)

        # Precompute pairwise scores
        pair_score, pair_sim, pair_depth = LayerEvaluator._compute_scores(ground_truth_layers, predicted_layers, P, G)

        # DP table and pointer
        dp, ptr = LayerEvaluator._build_dp_table(P, G, pair_score)

        # Backtrack to recover mapping
        mapping = LayerEvaluator._get_mapping(P, G, pair_score, pair_sim, pair_depth, ptr)

        matching_score = dp[P][G] / (2.0 * max(P, G))  # maximum is 1.
        return matching_score, mapping

    @staticmethod
    def _compute_scores(
        ground_truth_layers: list[dict[str, Any]], preds: list[Layer], P: int, G: int
    ) -> tuple[list[list[float]], list[list[float]], list[list[bool]]]:
        """Compute pairwise scores between predicted and ground truth layers.

        For each pair of predicted and ground truth layers, computes three scores:
        1. Total score (pair_score): Sum of text similarity and depth matching scores
           - Text similarity: Levenshtein ratio between parsed material descriptions (0.0 to 1.0)
           - Depth score: Up to 1.0 (0.5 for matching start, 0.5 for matching end)
           Total score range is 0.0 to 2.0 per pair

        2. Similarity score (pair_sim): Just the text similarity using Levenshtein ratio

        3. Depth flag (pair_depth): Boolean indicating perfect depth match
           True only if both start and end depths match exactly

        Args:
            ground_truth_layers (list[dict[str, Any]]): The ground truth layers with
                material_description and depth_interval fields.
            preds (list[Layer]): The predicted layers to evaluate.
            P (int): The number of predicted layers.
            G (int): The number of ground truth layers.

        Returns:
            tuple: A tuple containing:
                - pair_score (list[list[float]]): Combined scores for each pred-gt pair.
                - pair_sim (list[list[float]]): Just the text similarity scores.
                - pair_depth (list[list[bool]]): Exact depth matching flags.
        """
        pair_score = [[0.0] * G for _ in range(P)]
        pair_sim = [[0.0] * G for _ in range(P)]
        pair_depth = [[False] * G for _ in range(P)]

        def compute_pair(pred_i: int, gt_j: int):
            pred_layer = preds[pred_i]
            gt_layer = ground_truth_layers[gt_j]
            parsed_text = parse_text(pred_layer.material_description.text)
            sim_score = Levenshtein.ratio(parsed_text, parse_text(gt_layer["material_description"]))
            depth_score = 0.0
            if pred_layer.depths is not None:
                if (pred_layer.depths.start is None and gt_layer["depth_interval"]["start"] == 0) or (
                    pred_layer.depths.start is not None
                    and pred_layer.depths.start.value == gt_layer["depth_interval"]["start"]
                ):
                    depth_score += 0.5
                if (
                    pred_layer.depths.end is not None
                    and pred_layer.depths.end.value == gt_layer["depth_interval"]["end"]
                ):
                    depth_score += 0.5
            return sim_score + depth_score, sim_score, depth_score == 1.0

        for i in range(P):
            for j in range(G):
                s, sim, d_ok = compute_pair(i, j)
                pair_score[i][j] = s
                pair_sim[i][j] = sim
                pair_depth[i][j] = d_ok
        return pair_score, pair_sim, pair_depth

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
        pair_score: list[list[float]],
        pair_sim: list[list[float]],
        pair_depth: list[list[bool]],
        ptr: list[list[str]],
    ) -> list[dict]:
        """Get the mapping between predicted and ground truth layers by backtracking through the DP table.

        Args:
            P (int): The number of predicted layers.
            G (int): The number of ground truth layers.
            pair_score (list[list[float]]): The pairwise scores between predicted and ground truth layers.
            pair_sim (list[list[float]]): The pairwise similarity scores between predicted and ground truth layers.
            pair_depth (list[list[bool]]): The pairwise depth correctness between predicted and ground truth layers.
            ptr (list[list[str]]): The pointer table for backtracking.

        Returns:
            list[dict]: The mapping between predicted and ground truth layers.
        """
        i, j = P, G
        mapping = []
        while i > 0 and j > 0:
            move = ptr[i][j]
            if move == "diag":
                pi, gj = i - 1, j - 1
                mapping.append(
                    {
                        "pred_idx": pi,
                        "gt_idx": gj,
                        "material_similarity": pair_sim[pi][gj],
                        "depth_ok": pair_depth[pi][gj],
                        "pair_score": pair_score[pi][gj],
                    }
                )
                i, j = i - 1, j - 1
            elif move == "up":
                i -= 1
            elif move == "left":
                j -= 1
            else:
                break
        mapping.reverse()
        return mapping

    @staticmethod
    def _update_prediction_flags(preds: list[Layer], mapping: list[dict]) -> None:
        """Update the prediction flags based on the mapping.

        Args:
            preds (list[Layer]): The list of predicted layers.
            mapping (list[dict]): The mapping between predicted and ground truth layers.

        Returns:
            None: The `preds` list is modified in place.
        """
        for pred in preds:
            pred.material_description.is_correct = False
            if pred.depths is not None:
                pred.depths.is_correct = False
            pred.is_correct = False
            pred.is_correct = None
            pred.matched_ground_truth_idx = None
        for m in mapping:
            pred = preds[m["pred_idx"]]
            pred.matched_ground_truth_idx = m["gt_idx"]
            pred.material_description.is_correct = (
                m["material_similarity"] >= MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
            )
            if pred.depths is not None:
                pred.depths.is_correct = m["depth_ok"]

            pred.is_correct = (
                pred.material_description.is_correct and pred.depths is not None and pred.depths.is_correct
            )

    # OLD
    @staticmethod
    def compute_borehole_affinity(
        ground_truth_layers: list[dict], predicted_layers: list[Layer]
    ) -> tuple[float, list[dict]]:
        """Computes the matching score between a prediction and a groundtruth borehole.

        This is relevant when there is more than one borehole per pdf. Computing this score allows to match the
        predictions identified in the document against the correct groundtruth. The matching score is computed by
        comparing the layers of each borehole identified to each layers in the groudtruth.

        Args:
            ground_truth_layers (list[dict]): list containing the ground truth for the layers
            predicted_layers (list[Layer]): object containing the list of the predicted layers

        Returns:
            tuple containing:
                - matching_score (float): a score that captures the similarity between the boreholes (1 is best, 0 is
                    worst)
                - mapping (list[dict]): a list of mappings between predicted and ground truth layers
        """
        unmatched_layers = ground_truth_layers.copy()
        matching_score = 0
        if not predicted_layers:
            return 0
        mapping = []
        for layer in predicted_layers:
            match, depth_interval_is_correct = LayerEvaluator.find_matching_layer(layer, unmatched_layers)
            if match:
                layer.material_description.is_correct = True
                if layer.depths is not None:
                    layer.depths.is_correct = depth_interval_is_correct
                layer.is_correct = depth_interval_is_correct
                mapping.append(
                    {
                        "pred_idx": predicted_layers.index(layer),
                        "gt_idx": ground_truth_layers.index(match),
                        "material_similarity": Levenshtein.ratio(
                            parse_text(layer.material_description.text), match["material_description"]
                        ),
                        "depth_ok": depth_interval_is_correct,
                    }
                )
            else:
                layer.material_description.is_correct = False
                layer.is_correct = False
                if layer.depths is not None:
                    layer.depths.is_correct = False
            matching_score += 1 if depth_interval_is_correct else 0
            matching_score += 1 if match else 0
        matching_score /= 2 * len(predicted_layers)
        return matching_score, mapping

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
        parsed_text = parse_text(layer.material_description.text)
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

            if layer.depths is not None and layer.depths.is_valid_depth_interval(start, end):
                unmatched_layers.remove(possible_match)
                return possible_match, True

        match = max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, x["material_description"]))
        unmatched_layers.remove(match)
        return match, False
