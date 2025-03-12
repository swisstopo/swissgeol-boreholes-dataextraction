"""Classes for evaluating the groundwater levels of a borehole."""

import logging
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy

import Levenshtein
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetrics
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.utility import _is_valid_depth_interval
from stratigraphy.layer.layer import Layer, LayersInBorehole
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9


class LayerEvaluator:
    """Class for evaluating the layer information of all boreholes in a document."""

    def __init__(
        self,
        layers_entries: dict[str : list[LayersInBorehole]],
        ground_truth: GroundTruth,
        gt_to_pred_matching: dict[str : dict[int:int]],
    ):
        """Initializes the LayerEvaluator object.

        Args:
            layers_entries (dict[str : list[LayersInBorehole]]): The layers to evaluate.
            ground_truth (GroundTruth): The ground truth.
            gt_to_pred_matching (dict[str : dict[int:int]]): the dict matching the index of the gt borehole to pred

            layers_entries (dict[str : list[LayersInBorehole]]): The layers to evaluate. The expected
                format is a dict with the filename as key, and the lists of all the Borehole layers identified for each
                profile as value. The Borehole layers are themself a list of Layers.
            ground_truth (GroundTruth): The ground truth.
            gt_to_pred_matching (dict[str : dict[int:int]]): The dict matching the index of the groundtruth borehole
                to the prediction. It is mostly relevant when there is multiple boreholes in a documents (else it is
                just {0:0}). There is one entry for each of the files.
        """
        self.ground_truth: GroundTruth = ground_truth
        self.layers_entries: dict[str : list[LayersInBorehole]] = layers_entries
        self.gt_to_pred_matching: dict[str : dict[int:int]] = gt_to_pred_matching

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

        # iteration over all the files
        for filename, layers_in_document in self.layers_entries.items():
            # the groundtruth matching the current file is fetched, so is the dict matching the boreholes in the
            # groundtruth to the ones in the prediction
            ground_truth_for_file = self.ground_truth.for_file(filename)
            gt_to_pred_index = self.gt_to_pred_matching[filename]

            hits_for_all_borehole = 0
            total_predictions_for_all_boreholes = 0
            fn_for_all_boreholes = 0

            # iteration over all the borehole present in the groundtruth of the file
            for gt_index, ground_truth_borehole in ground_truth_for_file.items():
                # the borehole that is the most similar to the ground truth is fetched
                pred_borehole_index = gt_to_pred_index.get(gt_index)
                if pred_borehole_index is None:
                    # When the data extraction pipeline finds fewer boreholes than there actually is in the ground
                    # truth, the matching is not one-to-one and we have to skip some boreholes in the evaluation.
                    continue
                borehole_layers = layers_in_document[pred_borehole_index]

                number_of_truth_values = len(ground_truth_borehole["layers"])
                hits = 0
                total_predictions = 0

                for layer in borehole_layers.layers:
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

            # at this point we have the global statistics for all the boreholes in the documents
            if total_predictions > 0:
                overall_metrics.metrics[filename] = Metrics(
                    tp=hits_for_all_borehole,
                    fp=total_predictions_for_all_boreholes - hits_for_all_borehole,
                    fn=fn_for_all_boreholes,
                )

        return overall_metrics

    @staticmethod
    def evaluate_borehole(borehole_layers: list[LayersInBorehole], all_ground_truth_layers: dict[int:list]):
        """Evaluate all predicted layers for a borehole against the ground truth.

        Also performs the matching groundtruth to prediction when there is more than one borehole in the document.
        It is for this reason that the layers are the first element that needs to be elaluated.

        Args:
            borehole_layers (list[LayersInBorehole]): The predicted layers for all the boreholes in the document.
            all_ground_truth_layers(dict[int:list]): The ground truth layers for all the borehole,
                with the borehole index mapping to the list of layers.

        Returns:
            matching_dict_gt_to_pred (dict) : a dict mapping each groundtruth to the most similar identifies borehole.
        """
        pred_vs_gt_matching_score = defaultdict(dict)
        # make a copy of borehole_layers to avoid modifying internal state during matching
        borehole_layers_copy = deepcopy(borehole_layers)
        for gt_idx, ground_truth_layers in all_ground_truth_layers.items():
            for pred_idx, predicted_layers in enumerate(borehole_layers_copy):
                # evaluation loop
                matching_score = LayerEvaluator.compute_borehole_affinity(ground_truth_layers, predicted_layers)
                pred_vs_gt_matching_score[gt_idx][pred_idx] = matching_score

        # matching
        matching_dict_gt_to_pred = dict()
        assigned_preds = set()
        while pred_vs_gt_matching_score:
            max_score = float("-inf")
            for gt_idx, pred_scores in pred_vs_gt_matching_score.items():
                for pred_idx, score in pred_scores.items():
                    if score > max_score and pred_idx not in assigned_preds:  # can't assign the same pred twice
                        max_score = score
                        best_matches = (gt_idx, pred_idx)

            gt_best_idx, pred_best_idx = best_matches
            matching_dict_gt_to_pred[gt_best_idx] = pred_best_idx
            assigned_preds.add(pred_best_idx)  # Mark this pred_idx as used

            # Remove the matched gt_idx from consideration
            del pred_vs_gt_matching_score[gt_best_idx]

            if len(assigned_preds) == len(borehole_layers_copy):
                # all preds have been assigned
                break

        # now compute the real statistics for the matched pairs
        for gt_idx, pred_idx in matching_dict_gt_to_pred.items():
            # This method makes an internal modification to borehole_layers
            LayerEvaluator.compute_borehole_affinity(all_ground_truth_layers[gt_idx], borehole_layers[pred_idx])
        return matching_dict_gt_to_pred

    @staticmethod
    def compute_borehole_affinity(ground_truth_layers: list[dict], predicted_layers: LayersInBorehole) -> float:
        """Computes the matching score between a prediction and a groundtruth borehole.

        This is relevant when there is more than one borehole per pdf. Computing this score allows to match the
        predictions identified in the document against the correct groundtruth. The matching score is computed by
        comparing the layers of each borehole identified to each layers in the groudtruth.

        Args:
            ground_truth_layers (list[dict]): list containing the ground truth for the layers
            predicted_layers (LayersInBorehole): object containing the list of the predicted layers

        Returns:
            matching_score (float): a score that captures the similarity between the boreholes (1 is best, 0 is worst)
        """
        unmatched_layers = ground_truth_layers.copy()
        matching_score = 0
        if not predicted_layers.layers:
            return 0
        for layer in predicted_layers.layers:
            match, depth_interval_is_correct = LayerEvaluator.find_matching_layer(layer, unmatched_layers)
            if match:
                layer.material_description.feature.is_correct = True
                layer.is_correct = depth_interval_is_correct
            else:
                layer.material_description.feature.is_correct = False
                layer.is_correct = None
            matching_score += 1 if depth_interval_is_correct else 0
            matching_score += 1 if match else 0
        matching_score /= 2 * len(predicted_layers.layers)
        return matching_score

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
