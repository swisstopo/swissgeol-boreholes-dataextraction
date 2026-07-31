"""Utility functions for evaluation."""

import dataclasses
from collections.abc import Callable
from typing import TypeVar

from core.benchmark_utils import Metrics


@dataclasses.dataclass
class EvaluationResults:
    """Class that contains that data about which extracted entries could be matched with which ground truth values."""

    extracted_correct: list[bool]
    ground_truth_correct: list[bool]
    summary: str | None = None

    @property
    def metrics(self) -> Metrics:
        """Derive the metrics (true positives, false positives, fales negatives) from the evaluation results."""
        tp = sum(1 for is_correct in self.extracted_correct if is_correct)
        return Metrics(
            tp=tp, fp=len(self.extracted_correct) - tp, fn=len(self.ground_truth_correct) - tp, summary=self.summary
        )


T = TypeVar("T")
U = TypeVar("U")


def evaluate_single(
    extracted: T | None,
    ground_truth: U | None,
    match: Callable[[T, U], bool],
    create_summary: Callable[[T], str] | None = None,
) -> EvaluationResults:
    """Count evaluation metrics by comparing an optional predicted value against an optional ground truth value.

    Args:
        extracted: (T | None): The predicted value to evaluate (if any).
        ground_truth (U | None): The ground truth value to compare against (if any).
        match (Callable[[T, U], bool]): A function that defines when the extracted value matches the ground truth
            value.
        create_summary (Callable[[T], str] | None): An optional function to summarize an extracted value as a string.

    Returns:
        EvaluationResults: The evaluation results, including true positive, false positive and false negative counts.
    """
    extracted_list = [extracted] if extracted is not None else []
    ground_truth_list = [ground_truth] if ground_truth is not None else []
    return evaluate(extracted_list, ground_truth_list, match, create_summary)


def evaluate(
    extracted: list[T],
    ground_truth: list[U],
    match: Callable[[T, U], bool],
    create_summary: Callable[[T], str] | None = None,
) -> EvaluationResults:
    """Count evaluation metrics by comparing predicted values against ground truth.

    Args:
        extracted: (list[T]): The predicted values to evaluate.
        ground_truth (list[U]): The ground truth values to compare against.
        match (Callable[[T, U], bool]): A function that defines when an extracted value matches a ground truth value.
        create_summary (Callable[[T], str] | None): An optional function to summarize an extracted value as a string.

    Returns:
        EvaluationResults: The evaluation results, including true positive, false positive and false negative counts.
    """
    extracted_correct = [False for _ in extracted]
    ground_truth_correct = [False for _ in ground_truth]
    for extracted_index, extracted_value in enumerate(extracted):
        matched_gt_index = next(
            (
                ground_truth_index
                for ground_truth_index, value in enumerate(ground_truth)
                if not ground_truth_correct[ground_truth_index] and match(extracted_value, value)
            ),
            None,
        )
        if matched_gt_index is not None:
            extracted_correct[extracted_index] = True
            ground_truth_correct[matched_gt_index] = True

    # join summary of multiple extracted values for a single borehole
    summary = None if create_summary is None else ";".join([create_summary(extracted) for extracted in extracted])

    return EvaluationResults(extracted_correct, ground_truth_correct, summary=summary)
