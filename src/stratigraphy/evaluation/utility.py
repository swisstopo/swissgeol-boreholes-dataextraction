"""Utility functions for evaluation."""

from collections import Counter

import Levenshtein
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.layer.layer import Layer
from stratigraphy.util.interval import Interval
from stratigraphy.util.util import parse_text

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9


def count_against_ground_truth(values: list[str], ground_truth: list[str]) -> Metrics:
    """Count evaluation metrics by comparing predicted values against ground truth.

    Metrics are calculated as follows:
    - True Positives (tp): Number of values that appear in both lists (counting duplicates)
    - False Positives (fp): Number of extra predictions (len(values) - tp)
    - False Negatives (fn): Number of missed ground truth values (len(ground_truth) - tp)

    Args:
        values (list[str]): The predicted values to evaluate
        ground_truth (list[str]): The ground truth values to compare against

    Returns:
        Metrics: Object containing tp, fp, and fn counts
    """
    # Counter deals with duplicates when doing intersection
    values_counter = Counter(values)
    ground_truth_counter = Counter(ground_truth)

    tp = (values_counter & ground_truth_counter).total()  # size of intersection
    return Metrics(tp=tp, fp=len(values) - tp, fn=len(ground_truth) - tp)


def _is_valid_depth_interval(depth_interval: Interval, start: float, end: float) -> bool:
    """Validate if the depth intervals match.

    Args:
        depth_interval (Interval): The depth interval to compare.
        start (float): The start value of the interval.
        end (float): The end value of the interval.

    Returns:
        bool: True if the depth intervals match, False otherwise.
    """
    if depth_interval is None:
        return False

    if depth_interval.start is None:
        return (start == 0) and (end == depth_interval.end.value)

    if (depth_interval.start is not None) and (depth_interval.end is not None):
        return start == depth_interval.start.value and end == depth_interval.end.value

    return False


def find_matching_layer(layer: Layer, unmatched_layers: list[dict]) -> tuple[dict, bool] | tuple[None, None]:
    """Find the matching layer in the ground truth.

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

        if _is_valid_depth_interval(layer.depth_interval, start, end):
            unmatched_layers.remove(possible_match)
            return possible_match, True

    match = max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, x["material_description"]))
    unmatched_layers.remove(match)
    return match, False
