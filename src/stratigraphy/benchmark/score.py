"""Evaluate the predictions against the ground truth."""

import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.util.predictions import FilePredictions
from stratigraphy.util.util import parse_text

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

logger = logging.getLogger(__name__)


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


def get_scores(
    predictions: dict, number_of_truth_values: dict, return_document_level_metrics: bool
) -> dict | tuple[dict, pd.DataFrame]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.
        return_document_level_metrics (bool): Whether to return the document level metrics.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the overall F1, precision and recall as a dictionary and the
        individual document metrics as a DataFrame.
    """
    document_level_metrics = {
        "document_name": [],
        "F1": [],
        "precision": [],
        "recall": [],
        "Depth_interval_accuracy": [],
        "Number Elements": [],
        "Number wrong elements": [],
    }
    # separate list to calculate the overall depth interval accuracy is required,
    # as the depth interval accuracy is not calculated for documents with no correct
    # material predictions.
    depth_interval_accuracies = []
    for filename, file_prediction in predictions.items():
        hits = 0
        depth_interval_hits = 0
        depth_interval_occurences = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                hits += 1
                if layer.depth_interval_is_correct:
                    depth_interval_hits += 1
                    depth_interval_occurences += 1
                elif layer.depth_interval_is_correct is not None:
                    depth_interval_occurences += 1

            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")
        tp = hits
        fp = len(file_prediction.layers) - tp
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
        try:
            document_level_metrics["Depth_interval_accuracy"].append(depth_interval_hits / depth_interval_occurences)
            depth_interval_accuracies.append(depth_interval_hits / depth_interval_occurences)
        except ZeroDivisionError:
            document_level_metrics["Depth_interval_accuracy"].append(None)

    if len(document_level_metrics["precision"]):
        overall_precision = sum(document_level_metrics["precision"]) / len(document_level_metrics["precision"])
        overall_recall = sum(document_level_metrics["recall"]) / len(document_level_metrics["recall"])
        try:
            overall_depth_interval_accuracy = sum(depth_interval_accuracies) / len(depth_interval_accuracies)
        except ZeroDivisionError:
            overall_depth_interval_accuracy = None
    else:
        overall_precision = 0
        overall_recall = 0

    if overall_depth_interval_accuracy is None:
        overall_depth_interval_accuracy = 0

    if return_document_level_metrics:
        return {
            "F1": f1(overall_precision, overall_recall),
            "precision": overall_precision,
            "recall": overall_recall,
            "depth_interval_accuracy": overall_depth_interval_accuracy,
        }, pd.DataFrame(document_level_metrics)
    else:
        return {
            "F1": f1(overall_precision, overall_recall),
            "precision": overall_precision,
            "recall": overall_recall,
            "depth_interval_accuracy": overall_depth_interval_accuracy,
        }


def evaluate_borehole_extraction(predictions: dict, number_of_truth_values: dict) -> tuple[dict, pd.DataFrame]:
    """Evaluate the borehole extraction predictions.

    Args:
       predictions (dict): The FilePredictions objects.
       number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the overall metrics as a dictionary and the
        individual document metrics as a DataFrame.
    """
    layer_metrics, layer_document_level_metrics = evaluate_layer_extraction(predictions, number_of_truth_values)
    coordinate_metrics, coordinate_document_level_metrics = evaluate_metadata(predictions)
    metrics = {**layer_metrics, **coordinate_metrics}
    document_level_metrics = pd.merge(
        layer_document_level_metrics, coordinate_document_level_metrics, on="document_name", how="outer"
    )
    return metrics, document_level_metrics


def evaluate_metadata(predictions: dict) -> tuple[dict, pd.DataFrame]:
    """Evaluate the metadata predictions.

    Args:
        predictions (dict): The FilePredictions objects.

    Returns:
        tuple[dict, pd.DataFrame]: The overall coordinate accuracy and the individual document metrics as a DataFrame.
    """
    document_level_metrics = {
        "document_name": [],
        "coordinates": [],
    }
    coordinates_tp = 0  # correct prediction
    coordinates_fn = 0  # no predictions, i.e. None
    coordinates_fp = 0  # wrong prediction
    # there is no such thing as true negatives as each document has coordinates - though not necessarily specified
    number_coordinate_predictions = 0
    for file_name, file_prediction in predictions.items():
        if file_prediction.metadata_is_correct["coordinates"]:
            coordinates_tp += 1
            number_coordinate_predictions += 1
            document_level_metrics["document_name"].append(file_name)
            document_level_metrics["coordinates"].append(1)
        elif file_prediction.metadata_is_correct["coordinates"] is None:
            coordinates_fn += 1
            number_coordinate_predictions += 1
        else:
            coordinates_fp += 1
            number_coordinate_predictions += 1
            document_level_metrics["document_name"].append(file_name)
            document_level_metrics["coordinates"].append(0)
    try:
        coordinate_precision = coordinates_tp / (coordinates_tp + coordinates_fp)
    except ZeroDivisionError:
        coordinate_precision = 0
    try:
        coordinate_recall = coordinates_tp / (coordinates_tp + coordinates_fn)
    except ZeroDivisionError:
        coordinate_recall = 0

    metrics = {
        "coordinate_accuracy": coordinates_tp / number_coordinate_predictions,
        "coordinate_precision": coordinate_precision,
        "coordinate_recall": coordinate_recall,
        "coordinate_f1": f1(coordinate_precision, coordinate_recall),
    }
    return metrics, pd.DataFrame(document_level_metrics)


def evaluate_layer_extraction(predictions: dict, number_of_truth_values: dict) -> tuple[dict, pd.DataFrame]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The FilePredictions objects.
        number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the overall F1, precision and recall as a dictionary and the
        individual document metrics as a DataFrame.
    """
    metrics = {}
    metrics["all"], document_level_metrics = get_scores(
        predictions, number_of_truth_values, return_document_level_metrics=True
    )
    # create predictions by language
    predictions_by_language = defaultdict(dict)
    for file_name, file_predictions in predictions.items():
        language = file_predictions.language
        predictions_by_language[language][file_name] = file_predictions

    for language, language_predictions in predictions_by_language.items():
        language_number_of_truth_values = {
            file_name: number_of_truth_values[file_name] for file_name in language_predictions
        }
        metrics[language] = get_scores(
            language_predictions, language_number_of_truth_values, return_document_level_metrics=False
        )

    logging.info("Macro avg:")
    logging.info(
        f"F1: {metrics['all']['F1']:.1%}, "
        f"precision: {metrics['all']['precision']:.1%}, recall: {metrics['all']['recall']:.1%}, "
        f"depth_interval_accuracy: {metrics['all']['depth_interval_accuracy']:.1%}"
    )

    _metrics = {}
    for language, language_metrics in metrics.items():
        for metric_type, value in language_metrics.items():
            if language == "all":
                _metrics[metric_type] = value
            else:
                _metrics[f"{language}_{metric_type}"] = value
    return _metrics, document_level_metrics


def create_predictions_objects(
    predictions: dict, ground_truth_path: Path | None
) -> tuple[dict[FilePredictions], dict]:
    """Create predictions objects from the predictions and evaluate them against the ground truth.

    Args:
        predictions (dict): The predictions from the predictions.json file.
        ground_truth_path (Path　| None): The path to the ground truth file.

    Returns:
        tuple[dict[FilePredictions], dict]: The predictions objects and the number of ground truth values per file.
    """
    if ground_truth_path and os.path.exists(ground_truth_path):  # for inference no ground truth is available
        ground_truth = GroundTruth(ground_truth_path)
        ground_truth_is_present = True
    else:
        logging.warning("Ground truth file not found.")
        ground_truth_is_present = False

    number_of_truth_values = {}
    predictions_objects = {}
    for file_name, file_predictions in predictions.items():
        prediction_object = FilePredictions.create_from_json(file_predictions, file_name)

        predictions_objects[file_name] = prediction_object
        if ground_truth_is_present:
            ground_truth_for_file = ground_truth.for_file(file_name)
            if ground_truth_for_file:
                predictions_objects[file_name].evaluate(ground_truth_for_file)
                number_of_truth_values[file_name] = len(ground_truth_for_file["layers"])

    return predictions_objects, number_of_truth_values


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
    metrics, document_level_metrics = evaluate_borehole_extraction(
        predictions_path, ground_truth_path, input_directory, out_directory
    )
