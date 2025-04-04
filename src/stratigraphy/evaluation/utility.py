"""Utility functions for evaluation."""

from collections import Counter

from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.layer.layer import LayerDepths


def count_against_ground_truth(values: list[any], ground_truth: list[any]) -> Metrics:
    """Count evaluation metrics by comparing predicted values against ground truth.

    Metrics are calculated as follows:
    - True Positives (tp): Number of values that appear in both lists (counting duplicates)
    - False Positives (fp): Number of extra predictions (len(values) - tp)
    - False Negatives (fn): Number of missed ground truth values (len(ground_truth) - tp)

    Args:
        values (list[any]): The predicted values to evaluate
        ground_truth (list[any]): The ground truth values to compare against

    Returns:
        Metrics: Object containing tp, fp, and fn counts
    """
    # Counter deals with duplicates when doing intersection
    values_counter = Counter(values)
    ground_truth_counter = Counter(ground_truth)

    tp = (values_counter & ground_truth_counter).total()  # size of intersection
    return Metrics(tp=tp, fp=len(values) - tp, fn=len(ground_truth) - tp)


def _is_valid_depth_interval(depths: LayerDepths, start: float, end: float) -> bool:
    """Validate if the depth intervals match.

    Args:
        depths (LayerDepths): The layer depths to compare.
        start (float): The start value of the interval.
        end (float): The end value of the interval.

    Returns:
        bool: True if the depth intervals match, False otherwise.
    """
    if depths is None:
        return False

    if depths.start is None:
        return (start == 0) and (end == depths.end.value)

    if (depths.start is not None) and (depths.end is not None):
        return start == depths.start.value and end == depths.end.value

    return False
