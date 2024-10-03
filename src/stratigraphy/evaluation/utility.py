"""Utility functions for evaluation."""

from collections import Counter

import Levenshtein
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.layer.layer import Layer
from stratigraphy.util.util import parse_text


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


def find_matching_layer(layer: Layer, unmatched_layers: list[dict]) -> tuple[dict, bool] | tuple[None, None]:
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
