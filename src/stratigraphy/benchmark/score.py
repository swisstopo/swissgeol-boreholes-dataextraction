import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from stratigraphy import DATAPATH
from stratigraphy.benchmark.ground_truth import GroundTruth

logger = logging.getLogger(__name__)


def f1(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def evaluate_matching(predictions_path: Path, ground_truth_path: Path) -> tuple[dict, pd.DataFrame]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions_path (Path): Path to the predictions.json file.
        ground_truth_path (Path): Path to the ground truth annotated data.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the overall F1, precision and recall as a dictionary and the individual document metrics as a DataFrame.
    """
    ground_truth = GroundTruth(ground_truth_path)
    with open(predictions_path, "r") as in_file:
        predictions = json.load(in_file)

    document_level_metrics = {
        "document_name": [],
        "F1": [],
        "precision": [],
        "recall": [],
        "Number Elements": [],
        "Number wrong elements": [],
    }
    for filename in predictions:
        if filename in ground_truth.ground_truth_descriptions:
            prediction_descriptions = [
                GroundTruth.parse(entry["description"]) for entry in predictions[filename]["layers"]
            ]
            prediction_descriptions = [description for description in prediction_descriptions if description]
            ground_truth_for_file = ground_truth.for_file(filename)

            hits = []
            for value in prediction_descriptions:
                if ground_truth_for_file.is_correct(value):
                    hits.append(value)
            tp = len(hits)
            fp = len(prediction_descriptions) - tp
            fn = len(ground_truth_for_file.descriptions) - tp

            if tp:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
            else:
                precision = 0
                recall = 0
            document_level_metrics["document_name"].append(filename)
            document_level_metrics["precision"].append(precision)
            document_level_metrics["recall"].append(recall)
            document_level_metrics["F1"].append(f1(precision, recall))
            document_level_metrics["Number Elements"].append(len(ground_truth_for_file.descriptions))
            document_level_metrics["Number wrong elements"].append(len(ground_truth_for_file.descriptions) - len(hits))

    if len(document_level_metrics["precision"]):
        overall_precision = sum(document_level_metrics["precision"]) / len(document_level_metrics["precision"])
        overall_recall = sum(document_level_metrics["recall"]) / len(document_level_metrics["recall"])
        logging.info("Macro avg:")
        logging.info(
            f"F1: {f1(overall_precision, overall_recall):.1%}, precision: {overall_precision:.1%}, recall: {overall_recall:.1%}"
        )

    worst_count = 5
    if len(document_level_metrics["precision"]) > worst_count:
        best_precisions = sorted(document_level_metrics["precision"])[worst_count:]
        best_recalls = sorted(document_level_metrics["recall"])[worst_count:]
        precision = sum(best_precisions) / len(best_precisions)
        recall = sum(best_recalls) / len(best_recalls)
        logger.info(f"Ignoring worst {worst_count}:")
        logger.info(f"F1: {f1(precision, recall):.1%}, precision: {precision:.1%}, recall: {recall:.1%}")

    return {
        "F1": f1(overall_precision, overall_recall),
        "precision": overall_precision,
        "recall": overall_recall,
    }, pd.DataFrame(document_level_metrics)


if __name__ == "__main__":
    predictions_path = DATAPATH / "Benchmark" / "extract" / "predictions.json"
    ground_truth_path = DATAPATH / "Benchmark" / "ground_truth.json"

    metrics = evaluate_matching(predictions_path, ground_truth_path)
