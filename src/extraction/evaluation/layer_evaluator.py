"""Classes for evaluating the layer and depth predictions of a borehole."""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import Levenshtein

from core.benchmark_utils import Metrics
from core.ground_truth import GroundTruthBorehole, GroundTruthLayer
from extraction.features.predictions.borehole_predictions import (
    BoreholePredictions,
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
BOREHOLE_ORDER_BONUS_WEIGHT = 0.4  # tune, revisit, if necessary


class LayerEvaluator:
    """Class for evaluating the layer information of all boreholes in a document."""

    @staticmethod
    def get_layer_metrics(file_predictions: FileLayersWithGroundTruth) -> Metrics:
        """Calculate layer-level metrics for the given file's predictions.

        Args:
            file_predictions (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: The computed layer metrics.
        """
        return LayerEvaluator.calculate_metrics(
            file_predictions=file_predictions,
            num_ground_truth_fn=lambda ground_truth_layers: len(ground_truth_layers),
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.is_correct,
        )

    @staticmethod
    def get_material_description_metrics(file_predictions: FileLayersWithGroundTruth) -> Metrics:
        """Calculate metrics for material description extraction across all boreholes in a file.

        Args:
            file_predictions (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: Aggregated material metrics across all boreholes.
        """

        def num_ground_truth_fn(ground_truth_layers: list[GroundTruthLayer]):
            return sum(lay.material_description is not None for lay in ground_truth_layers)

        return LayerEvaluator.calculate_metrics(
            file_predictions=file_predictions,
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: layer.description_nonempty(),
            per_layer_condition=lambda layer: layer.material_description.is_correct,
        )

    @staticmethod
    def get_depth_interval_metrics(file_predictions: FileLayersWithGroundTruth) -> Metrics:
        """Calculate metrics for depth interval extraction across all boreholes in a file.

        Args:
            file_predictions (FileLayersWithGroundTruth): Layer predictions paired with ground truth.

        Returns:
            Metrics: Aggregated depth metrics across all boreholes.
        """

        def num_ground_truth_fn(ground_truth_layers: list[GroundTruthLayer]):
            return sum(lay.depth_interval is not None for lay in ground_truth_layers)

        return LayerEvaluator.calculate_metrics(
            file_predictions=file_predictions,
            num_ground_truth_fn=num_ground_truth_fn,
            per_layer_filter=lambda layer: True,
            per_layer_condition=lambda layer: layer.depths is not None and layer.depths.is_correct,
        )

    @staticmethod
    def calculate_metrics(
        file_predictions: FileLayersWithGroundTruth,
        num_ground_truth_fn: Callable[[list[GroundTruthLayer]], int],
        per_layer_filter: Callable[[Layer], bool],
        per_layer_condition: Callable[[Layer], bool],
    ) -> Metrics:
        """Calculate metrics based on a condition per layer, after applying a filter.

        Args:
            file_predictions (FileLayersWithGroundTruth): Borehole layer predictions paired with ground truth.
            num_ground_truth_fn (Callable[[list[GroundTruthLayer]], int]): Function that returns the number of
                ground truth.
            per_layer_filter (Callable[[Layer], bool]): Function to filter layers to consider.
            per_layer_condition (Callable[[Layer], bool]): Function that returns True if the layer is a hit.

        Returns:
            Metrics: The calculated metrics.
        """
        hits_for_all_borehole = 0
        total_predictions_for_all_boreholes = 0
        fn_for_all_boreholes = 0

        for borehole_data in file_predictions.boreholes:
            number_of_truth_values = num_ground_truth_fn(borehole_data.ground_truth)
            tp = 0
            total_predictions = 0

            layers = borehole_data.layers.layers if borehole_data.layers else []
            for layer in layers:
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

        def set_material_description_flag(predicted_layer, ground_truth_layers):
            predicted_layer.material_description.is_correct = (
                score_material_descriptions(predicted_layer, ground_truth_layers)
                >= MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
            )

        def set_layer_flag(predicted_layer, ground_truth_layers):
            predicted_layer.is_correct = (
                score_depths(predicted_layer, ground_truth_layers) == MAX_DEPTH_SCORE
                and score_material_descriptions(predicted_layer, ground_truth_layers)
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
                # Only consider predicted layers with a description when scoring the material descriptions
                LayerEvaluator.apply_mapping(
                    borehole_data.ground_truth,
                    [layer for layer in predicted_layers if layer.description_nonempty()],
                    score_material_descriptions,
                    set_material_description_flag,
                )
                LayerEvaluator.apply_mapping(borehole_data.ground_truth, predicted_layers, score_layer, set_layer_flag)

        layer_metrics = LayerEvaluator.get_layer_metrics(file_predictions)
        depth_interval_metrics = LayerEvaluator.get_depth_interval_metrics(file_predictions)
        material_description_metrics = LayerEvaluator.get_material_description_metrics(file_predictions)

        return layer_metrics, depth_interval_metrics, material_description_metrics

    @staticmethod
    def apply_mapping(
        ground_truth_layers: list[GroundTruthLayer],
        predicted_layers: list[Layer],
        scoring_fn: Callable[[Layer, GroundTruthLayer], float],
        set_flag_fn: Callable[[Layer, GroundTruthLayer], None],
    ) -> None:
        """Apply a scoring function to map ground truth layers to predicted layers and set flags."""
        _, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(
            ground_truth_layers, predicted_layers, scoring_fn
        )
        for predicted_layer, ground_truth_layer in mapping:
            set_flag_fn(predicted_layer, ground_truth_layer)

    @staticmethod
    def match_boreholes_to_ground_truth(
        file_predictions: FilePredictions, ground_truth_for_file: list[GroundTruthBorehole]
    ) -> list[BoreholePredictionsWithGroundTruth]:
        """Match predicted boreholes to ground truth boreholes.

        This method compares the predicted boreholes with the ground truth boreholes and establishes a mapping
            between them based on content similarity, plus an additive bonus (weighted by
            `BOREHOLE_ORDER_BONUS_WEIGHT`) for agreeing on page position (reading order). The bonus applies to
            every candidate pair, not only near-ties, since ground truth boreholes are usually, but not always,
            annotated in reading order.

        Args:
            file_predictions (FilePredictions): all predictions for the file
            ground_truth_for_file (list[GroundTruthBorehole]): the ground truth for the file

        Returns:
            list[BoreholePredictionsWithGroundTruth]: A list of matched borehole predictions with their ground truth.
        """
        predictions = file_predictions.borehole_predictions_list
        reading_rank = {id(pred): rank for rank, pred in enumerate(_boreholes_in_reading_order(predictions))}
        # using position as a tie-breaker when there are multiple boreholes in a file
        bonus_weight = BOREHOLE_ORDER_BONUS_WEIGHT if len(predictions) > 1 and len(ground_truth_for_file) > 1 else 0.0

        def relative_position(rank: int, count: int) -> float:
            return rank / (count - 1) if count > 1 else 0.0

        pred_positions = [relative_position(reading_rank[id(pred)], len(predictions)) for pred in predictions]

        pred_vs_gt_matching_score = defaultdict(dict)
        for gt_idx, ground_truth_borehole in enumerate(ground_truth_for_file):
            gt_position = relative_position(gt_idx, len(ground_truth_for_file))
            for pred_idx, pred in enumerate(predictions):
                content_score, _ = LayerEvaluator.compute_borehole_affinity_and_mapping(
                    ground_truth_borehole.layers, pred.layers_in_borehole.layers, score_layer
                )
                order_bonus = bonus_weight * (1 - abs(pred_positions[pred_idx] - gt_position))
                pred_vs_gt_matching_score[gt_idx][pred_idx] = content_score + order_bonus

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
                BoreholePredictionsWithGroundTruth(predictions[pred_best_idx], ground_truth_for_file[gt_best_idx])
            )
            assigned_preds.add(pred_best_idx)  # Mark this pred_idx as used

            # Remove the matched gt_idx from consideration
            del pred_vs_gt_matching_score[gt_best_idx]

            if len(assigned_preds) == len(predictions):
                # all preds have been assigned
                break

        # add entries with missing predictions for all unmatched ground truth boreholes (will count as false negatives)
        for gt_idx in pred_vs_gt_matching_score:
            matched_boreholes.append(
                BoreholePredictionsWithGroundTruth(predictions=None, ground_truth=ground_truth_for_file[gt_idx])
            )

        # add entries with missing ground truth for all unmatched prediction boreholes (will count as false positives)
        for pred_idx, pred in enumerate(predictions):
            if pred_idx not in assigned_preds:
                matched_boreholes.append(BoreholePredictionsWithGroundTruth(predictions=pred, ground_truth=None))
        return matched_boreholes

    @staticmethod
    def compute_borehole_affinity_and_mapping(
        ground_truth_layers: list[GroundTruthLayer],
        predicted_layers: list[Layer],
        scoring_fn: Callable[[Layer, GroundTruthLayer], float],
    ) -> tuple[float, list[tuple[Layer, GroundTruthLayer]]]:
        """Computes the matching score between a prediction and a groundtruth borehole.

        Computing this score allows to match the predictions identified in the document against the correct
        groundtruth. The matching score is computed by comparing the layers of each borehole identified to each
        layers in the ground truth.

        Args:
            ground_truth_layers (list[GroundTruthLayer]): list containing the ground truth for the layers
            predicted_layers (list[Layer]): object containing the list of the predicted layers
            scoring_fn (Callable[[Layer, GroundTruthLayer], float]): scoring function used for selecting best mapping

        Returns:
            tuple: containing
                - matching_score (float): a score that captures the similarity between the predicted and ground
                    truth layers. Maximum is 1.0.
                - mapping (list[(Layer, GroundTruthLayer)]): mappings between predicted and ground truth layers.
        """
        dp = PredToGroundTruthLayerDP(predicted_layers, ground_truth_layers, [0.0] * len(ground_truth_layers))
        return dp.solve(scoring_fn)


def score_material_descriptions(layer: Layer, ground_truth: GroundTruthLayer) -> float:
    """Scores how well the extracted material description matches the ground truth on a scale from 0 to 1."""
    parsed_text = parse_text(layer.material_description.text)
    return Levenshtein.ratio(parsed_text, parse_text(ground_truth.material_description))


def score_depths(layer: Layer, ground_truth: GroundTruthLayer) -> float:
    """Scores how well the extracted depths match the ground truth on a scale from 0 to 1.

    The total score is composed of 0.5 for matching start and 0.5 for matching end.
    """
    depth_score = 0.0
    ground_truth_start = ground_truth.depth_interval.start
    ground_truth_end = ground_truth.depth_interval.end

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


def score_layer(layer: Layer, ground_truth: GroundTruthLayer) -> float:
    """Scores how well the full layer matches the ground truth on a scale from 0 to 1."""
    return (score_material_descriptions(layer, ground_truth) + score_depths(layer, ground_truth)) / 2


class _BoreholePosition(NamedTuple):
    """Position of a borehole's first page, used to order it by reading position."""

    page: int
    y0: float
    y1: float
    x0: float


def _borehole_position(prediction: BoreholePredictions) -> _BoreholePosition:
    """Return the position of a borehole's first page, used to order it by reading position."""
    if not prediction.bounding_boxes:
        logger.warning(
            "Borehole %s has no bounding boxes; reading-order position defaults to page 0, top-left.",
            prediction.borehole_index,
        )
        return _BoreholePosition(0, 0.0, 0.0, 0.0)
    rect = prediction.bounding_boxes[0].get_outer_rect()
    return _BoreholePosition(prediction.bounding_boxes[0].page, rect.y0, rect.y1, rect.x0)


def _boreholes_in_reading_order(predictions: list[BoreholePredictions]) -> list[BoreholePredictions]:
    """Sort boreholes into page reading order: top-to-bottom by row, then left-to-right within a row."""
    positioned = sorted(
        ((_borehole_position(prediction), prediction) for prediction in predictions),
        key=lambda item: (item[0].page, item[0].y0),
    )

    # positioned is sorted by (page, y0)
    rows = []
    current_page, row_bottom = None, None
    for position, prediction in positioned:
        page, y0, y1 = position.page, position.y0, position.y1
        if page != current_page or y0 >= row_bottom:
            rows.append([])
            current_page, row_bottom = page, y1
        else:
            row_bottom = max(row_bottom, y1)
        rows[-1].append((position, prediction))

    return [prediction for row in rows for _, prediction in sorted(row, key=lambda item: item[0].x0)]
