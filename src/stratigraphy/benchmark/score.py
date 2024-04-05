"""Evaluate the predictions against the ground truth."""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.util.util import parse_text

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

logger = logging.getLogger(__name__)


def _get_from_all_pages(predictions: dict) -> dict:
    """Get all predictions from all pages.

    Args:
        predictions (dict): The predictions.

    Returns:
        dict: The predictions from all pages.
    """
    all_predictions = {}
    layers = []
    depths_materials_column_pairs = []
    for page in predictions:
        layers.extend(predictions[page]["layers"])
        depths_materials_column_pairs.extend(predictions[page]["depths_materials_column_pairs"])
    all_predictions["layers"] = layers
    if len(depths_materials_column_pairs):
        all_predictions["depths_materials_column_pairs"] = depths_materials_column_pairs
    return all_predictions


def f1(precision: float, recall: float) -> float:
    """Calculate the F1 score.

    Args:
        precision (float): Precision.
        recall (float): Recall.

    Returns:
        float: The F1 score.
    """
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def evaluate_matching(predictions: dict, number_of_truth_values: dict) -> tuple[dict, pd.DataFrame]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the overall F1, precision and recall as a dictionary and the
        individual document metrics as a DataFrame.
    """
    document_level_metrics = {
        "document_name": [],
        "F1": [],
        "precision": [],
        "recall": [],
        "Number Elements": [],
        "Number wrong elements": [],
    }
    for filename in predictions:
        all_predictions = _get_from_all_pages(predictions[filename])
        hits = 0
        for value in all_predictions["layers"]:
            if value["material_description"]["is_correct"]:
                hits += 1
            if parse_text(value["material_description"]["text"]) == "":
                print("Empty string found in predictions")
        tp = hits
        fp = len(all_predictions["layers"]) - tp
        fn = number_of_truth_values[filename] - tp

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
        document_level_metrics["Number Elements"].append(number_of_truth_values[filename])
        document_level_metrics["Number wrong elements"].append(fn + fp)

    if len(document_level_metrics["precision"]):
        overall_precision = sum(document_level_metrics["precision"]) / len(document_level_metrics["precision"])
        overall_recall = sum(document_level_metrics["recall"]) / len(document_level_metrics["recall"])
    else:
        overall_precision = 0
        overall_recall = 0

    logging.info("Macro avg:")
    logging.info(
        f"F1: {f1(overall_precision, overall_recall):.1%},"
        f"precision: {overall_precision:.1%}, recall: {overall_recall:.1%}"
    )

    return {
        "F1": f1(overall_precision, overall_recall),
        "precision": overall_precision,
        "recall": overall_recall,
    }, pd.DataFrame(document_level_metrics)


def add_ground_truth_to_predictions(predictions: dict, ground_truth_path: Path) -> tuple[dict, dict]:
    """Add the ground truth to the predictions.

    Args:
        predictions (dict): The predictions.
        ground_truth_path (Path): The path to the ground truth file.

    Returns:
        tuple[dict, dict]: The predictions with the ground truth added, and the number of ground truth values per file.
    """
    ground_truth = GroundTruth(ground_truth_path)

    number_of_truth_values = {}
    for file, file_predictions in predictions.items():
        ground_truth_for_file = ground_truth.for_file(file)
        number_of_truth_values[file] = len(ground_truth_for_file.descriptions)
        for page in file_predictions:
            for layer in file_predictions[page]["layers"]:
                layer["material_description"]["is_correct"] = ground_truth_for_file.is_correct(
                    layer["material_description"]["text"]
                )
    return predictions, number_of_truth_values


if __name__ == "__main__":
    # setup mlflow tracking; should be started before any other code
    # such that tracking is enabled in other parts of the code.
    # This does not create any scores, but will logg all the created images to mlflow.
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()

    # instantiate all paths
    input_directory = DATAPATH / "Benchmark"
    ground_truth_path = input_directory / "ground_truth.json"
    out_directory = input_directory / "evaluation"
    predictions_path = input_directory / "extract" / "predictions.json"

    # evaluate the predictions
    metrics, document_level_metrics = evaluate_matching(
        predictions_path, ground_truth_path, input_directory, out_directory
    )
