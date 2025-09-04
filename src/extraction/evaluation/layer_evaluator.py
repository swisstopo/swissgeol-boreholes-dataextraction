"""Classes for evaluating the groundwater levels of a borehole."""

import logging
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import Levenshtein

from extraction.evaluation.benchmark.metrics import OverallMetrics
from extraction.evaluation.evaluation_dataclasses import Metrics
from extraction.features.predictions.borehole_predictions import (
    BoreholePredictionsWithGroundTruth,
    FileLayersWithGroundTruth,
)
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.stratigraphy.layer.layer import Layer, LayersInBorehole
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
        """Calculate metrics for layer predictions."""

        def per_layer_action(layer: Layer):
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")

        return self.calculate_metrics(
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.material_description.is_correct,
            per_layer_action=per_layer_action,
        )

    def get_depth_interval_metrics(self) -> OverallMetrics:
        """Calculate metrics for depth interval predictions."""
        return self.calculate_metrics(
            per_layer_filter=lambda layer: layer.material_description.is_correct and layer.is_correct is not None,
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

        # iteration over all the files
        for file in self.file_layers_list:
            hits_for_all_borehole = 0
            total_predictions_for_all_boreholes = 0
            fn_for_all_boreholes = 0

            for borehole_data in file.boreholes:
                number_of_truth_values = len(borehole_data.ground_truth)
                hits = 0
                total_predictions = 0

                layers = borehole_data.layers.layers if borehole_data.layers else []
                for layer in layers:
                    if per_layer_action:
                        per_layer_action(layer)
                    if per_layer_filter(layer):
                        total_predictions += 1
                        if per_layer_condition(layer):
                            hits += 1

                fn = number_of_truth_values - hits

                hits_for_all_borehole += hits
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
        borehole_layers = [bh.layers_in_borehole for bh in file_predictions.borehole_predictions_list]
        all_ground_truth_layers = {
            idx: borehole_data["layers"] for idx, borehole_data in ground_truth_for_file.items()
        }

        pred_vs_gt_matching_score = defaultdict(dict)
        # make a copy of borehole_layers to avoid modifying internal state during matching
        borehole_layers_copy = deepcopy(borehole_layers)
        for gt_idx, ground_truth_layers in all_ground_truth_layers.items():
            for pred_idx, predicted_layers in enumerate(borehole_layers_copy):
                # evaluation loop
                matching_score, _ = LayerEvaluator.compute_borehole_affinity_and_mapping(
                    ground_truth_layers, predicted_layers
                )
                pred_vs_gt_matching_score[gt_idx][pred_idx] = matching_score

        # matching of all the boreholes detected to a borehole in the ground truth
        result = []
        assigned_preds = set()
        while pred_vs_gt_matching_score:
            max_score = float("-inf")
            for gt_idx, pred_scores in pred_vs_gt_matching_score.items():
                for pred_idx, score in pred_scores.items():
                    if score > max_score and pred_idx not in assigned_preds:  # can't assign the same pred twice
                        max_score = score
                        best_matches = (gt_idx, pred_idx)

            gt_best_idx, pred_best_idx = best_matches
            result.append(
                BoreholePredictionsWithGroundTruth(
                    file_predictions.borehole_predictions_list[pred_best_idx], ground_truth_for_file[gt_best_idx]
                )
            )
            assigned_preds.add(pred_best_idx)  # Mark this pred_idx as used

            # Remove the matched gt_idx from consideration
            del pred_vs_gt_matching_score[gt_best_idx]

            if len(assigned_preds) == len(borehole_layers_copy):
                # all preds have been assigned
                break

        # add entries with missing predictions for all unmatched ground truth boreholes (will count as false negatives)
        for gt_idx in pred_vs_gt_matching_score:
            result.append(
                BoreholePredictionsWithGroundTruth(predictions=None, ground_truth=ground_truth_for_file[gt_idx])
            )

        # add entries with missing ground truth for all unmatched prediction boreholes (will count as false positives)
        for index, pred in enumerate(file_predictions.borehole_predictions_list):
            if index not in assigned_preds:
                result.append(BoreholePredictionsWithGroundTruth(predictions=pred, ground_truth={}))

        # now compute the real statistics for the matched pairs of boreholes
        for borehole_data in result:
            if borehole_data.predictions:
                # This method makes an internal modification to borehole_layers
                LayerEvaluator.compute_borehole_affinity_and_mapping(
                    borehole_data.ground_truth.get("layers", []), borehole_data.predictions.layers_in_borehole
                )
        return result

    @staticmethod
    def compute_borehole_affinity_and_mapping(
        ground_truth_layers: list[dict[str, Any]],
        predicted_layers: "LayersInBorehole",
    ) -> tuple[float, list[dict[str, Any]]]:
        preds = predicted_layers.layers
        if not preds or not ground_truth_layers:
            return 0.0, []

        P, G = len(preds), len(ground_truth_layers)
        ground_truth_layers = sorted(
            ground_truth_layers, key=lambda gt: (gt["depth_interval"]["start"], gt["depth_interval"]["end"])
        )

        # Precompute pairwise scores
        pair_score, pair_sim, pair_depth = LayerEvaluator._compute_scores(ground_truth_layers, preds, P, G)

        # DP table and pointer
        dp, ptr = LayerEvaluator._build_dp_table(P, G, pair_score)

        # Backtrack to recover mapping
        mapping = LayerEvaluator._get_mapping(P, G, pair_score, pair_sim, pair_depth, ptr)

        # update predicted layer flags
        LayerEvaluator._update_prediction_flags(preds, mapping)

        matching_score = dp[P][G] / 2.0  # maximum is one point per layer
        return matching_score, mapping

    @staticmethod
    def _compute_scores(ground_truth_layers, preds, P, G):
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
    def _build_dp_table(P, G, pair_score):
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
    def _get_mapping(P, G, pair_score, pair_sim, pair_depth, ptr):
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
    def _update_prediction_flags(preds, mapping):
        for pred in preds:
            pred.material_description.is_correct = False
            # if pred.depths is not None:
            #     pred.depths.is_correct = False
            # pred.is_correct = False
            pred.is_correct = None
        for m in mapping:
            pi: int = m["pred_idx"]
            layer = preds[pi]

            layer.material_description.is_correct = True
            layer.is_correct = m["depth_ok"]
            # pi: int = m["pred_idx"]
            # preds[pi].material_description.is_correct = (
            #     m["material_similarity"] >= MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
            # )
            # if preds[pi].depths is not None:
            #     preds[pi].depths.is_correct = m["depth_ok"]

            # preds[pi].is_correct = (
            #     preds[pi].material_description.is_correct
            #     and preds[pi].depths is not None
            #     and preds[pi].depths.is_correct
            # )
