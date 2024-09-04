"""Evaluate the predictions against the ground truth."""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import DatasetMetrics, Metrics
from stratigraphy.util.draw import draw_predictions
from stratigraphy.util.predictions import FilePredictions
from stratigraphy.util.util import parse_text

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_scores(predictions: dict, number_of_truth_values: dict) -> tuple[DatasetMetrics, DatasetMetrics]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        tuple[DatasetMetrics, DatasetMetrics]: the metrics for the layers and depth intervals respectively
    """
    # separate list to calculate the overall depth interval accuracy is required,
    # as the depth interval accuracy is not calculated for documents with no correct
    # material predictions.
    depth_interval_metrics = DatasetMetrics()
    layer_metrics = DatasetMetrics()

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
        layer_metrics.metrics[filename] = Metrics(
            tp=hits, fp=len(file_prediction.layers) - hits, fn=number_of_truth_values[filename] - hits
        )
        if depth_interval_occurences > 0:
            depth_interval_metrics.metrics[filename] = Metrics(
                tp=depth_interval_hits, fp=depth_interval_occurences - depth_interval_hits, fn=0
            )

    return layer_metrics, depth_interval_metrics


def evaluate_borehole_extraction(
    predictions: dict[str, FilePredictions], number_of_truth_values: dict
) -> dict[str, DatasetMetrics]:
    """Evaluate the borehole extraction predictions.

    Args:
       predictions (dict): The FilePredictions objects.
       number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        dict[str, DatasetMetrics]: A dictionary that maps a metrics name to the corresponding DatasetMetrics object
    """
    all_metrics = evaluate_layer_extraction(predictions, number_of_truth_values)
    all_metrics["coordinates"] = get_metrics(predictions, "metadata_is_correct", "coordinates")
    all_metrics["elevation"] = get_metrics(predictions, "metadata_is_correct", "elevation")
    all_metrics["groundwater"] = get_metrics(predictions, "groundwater_is_correct", "groundwater")
    all_metrics["groundwater_depth"] = get_metrics(predictions, "groundwater_is_correct", "groundwater_depth")
    return all_metrics


def get_metrics(predictions: dict[str, FilePredictions], field_key: str, field_name: str) -> DatasetMetrics:
    """Get the metrics for a specific field in the predictions.

    Args:
        predictions (dict): The FilePredictions objects.
        field_key (str): The key to access the specific field in the prediction objects.
        field_name (str): The name of the field being evaluated.

    Returns:
        dict: The requested DatasetMetrics object.
    """
    dataset_metrics = DatasetMetrics()

    for file_name, file_prediction in predictions.items():
        dataset_metrics.metrics[file_name] = getattr(file_prediction, field_key)[field_name]

    return dataset_metrics


def evaluate_layer_extraction(predictions: dict, number_of_truth_values: dict) -> dict[str, DatasetMetrics]:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The FilePredictions objects.
        number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        dict[str, DatasetMetrics]: A dictionary that maps a metrics name to the corresponding DatasetMetrics object
    """
    all_metrics = {}
    all_metrics["layer"], all_metrics["depth_interval"] = get_scores(predictions, number_of_truth_values)

    # create predictions by language
    predictions_by_language = {"de": {}, "fr": {}}
    for file_name, file_predictions in predictions.items():
        language = file_predictions.language
        if language in predictions_by_language:
            predictions_by_language[language][file_name] = file_predictions

    for language, language_predictions in predictions_by_language.items():
        language_number_of_truth_values = {
            file_name: number_of_truth_values[file_name] for file_name in language_predictions
        }
        all_metrics[f"{language}_layer"], all_metrics[f"{language}_depth_interval"] = get_scores(
            language_predictions, language_number_of_truth_values
        )

    logging.info("Macro avg:")
    logging.info(
        f"F1: {all_metrics['layer'].macro_f1():.1%}, "
        f"precision: {all_metrics['layer'].macro_precision():.1%}, recall: {all_metrics['layer'].macro_recall():.1%}, "
        f"depth_interval_accuracy: {all_metrics['depth_interval'].macro_precision():.1%}"
    )

    return all_metrics


def create_predictions_objects(
    predictions: dict,
    ground_truth_path: Path | None,
) -> tuple[dict[str, FilePredictions], dict]:
    """Create predictions objects from the predictions and evaluate them against the ground truth.

    Args:
        predictions (dict): The predictions from the predictions.json file.
        ground_truth_path (Path　| None): The path to the ground truth file.

    Returns:
        tuple[dict[str, FilePredictions], dict]: The predictions objects and the number of ground truth values per
                                                 file.
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


def evaluate(
    predictions,
    ground_truth_path: Path,
    temp_directory: Path,
    input_directory: Path | None,
    draw_directory: Path | None,
):
    """Computes all the metrics, logs them, and creates corresponding MLFlow artifacts (when enabled)."""
    predictions, number_of_truth_values = create_predictions_objects(predictions, ground_truth_path)

    if input_directory and draw_directory:
        draw_predictions(predictions, input_directory, draw_directory)

    if number_of_truth_values:  # only evaluate if ground truth is available
        metrics = evaluate_borehole_extraction(predictions, number_of_truth_values)

        all_series = [
            metrics["layer"].to_dataframe("F1", lambda metric: metric.f1),
            metrics["layer"].to_dataframe("precision", lambda metric: metric.precision),
            metrics["layer"].to_dataframe("recall", lambda metric: metric.recall),
            metrics["depth_interval"].to_dataframe("Depth_interval_accuracy", lambda metric: metric.precision),
            metrics["layer"].to_dataframe("Number Elements", lambda metric: metric.tp + metric.fn),
            metrics["layer"].to_dataframe("Number wrong elements", lambda metric: metric.fp + metric.fn),
            metrics["coordinates"].to_dataframe("coordinates", lambda metric: metric.f1),
            metrics["elevation"].to_dataframe("elevation", lambda metric: metric.f1),
            metrics["groundwater"].to_dataframe("groundwater", lambda metric: metric.f1),
            metrics["groundwater_depth"].to_dataframe("groundwater_depth", lambda metric: metric.f1),
        ]
        document_level_metrics = pd.DataFrame()
        for series in all_series:
            document_level_metrics = document_level_metrics.join(series, how="outer")

        document_level_metrics.to_csv(
            temp_directory / "document_level_metrics.csv", index_label="document_name"
        )  # mlflow.log_artifact expects a file

        coordinates_metrics = metrics["coordinates"].overall_metrics()
        groundwater_metrics = metrics["groundwater"].overall_metrics()
        groundwater_depth_metrics = metrics["groundwater_depth"].overall_metrics()
        elevation_metrics = metrics["elevation"].overall_metrics()

        metrics_dict = {
            "F1": metrics["layer"].pseudo_macro_f1(),
            "recall": metrics["layer"].macro_recall(),
            "precision": metrics["layer"].macro_precision(),
            "depth_interval_accuracy": metrics["depth_interval"].macro_precision(),
            "de_F1": metrics["de_layer"].pseudo_macro_f1(),
            "de_recall": metrics["de_layer"].macro_recall(),
            "de_precision": metrics["de_layer"].macro_precision(),
            "de_depth_interval_accuracy": metrics["de_depth_interval"].macro_precision(),
            "fr_F1": metrics["fr_layer"].pseudo_macro_f1(),
            "fr_recall": metrics["fr_layer"].macro_recall(),
            "fr_precision": metrics["fr_layer"].macro_precision(),
            "fr_depth_interval_accuracy": metrics["fr_depth_interval"].macro_precision(),
            "coordinate_f1": coordinates_metrics.f1,
            "coordinate_recall": coordinates_metrics.recall,
            "coordinate_precision": coordinates_metrics.precision,
            "groundwater_f1": groundwater_metrics.f1,
            "groundwater_recall": groundwater_metrics.recall,
            "groundwater_precision": groundwater_metrics.precision,
            "groundwater_depth_f1": groundwater_depth_metrics.f1,
            "groundwater_depth_recall": groundwater_depth_metrics.recall,
            "groundwater_depth_precision": groundwater_depth_metrics.precision,
            "elevation_f1": elevation_metrics.f1,
            "elevation_recall": elevation_metrics.recall,
            "elevation_precision": elevation_metrics.precision,
        }

        # print the metrics
        logger.info("Performance metrics: %s", metrics_dict)

        if mlflow_tracking:
            import mlflow

            mlflow.log_metrics(metrics_dict)
            mlflow.log_artifact(temp_directory / "document_level_metrics.csv")
    else:
        logger.warning("Ground truth file not found. Skipping evaluation.")


if __name__ == "__main__":
    # setup mlflow tracking; should be started before any other code
    # such that tracking is enabled in other parts of the code.
    # This does not create any scores, but will log all the created images to mlflow.
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()

    # TODO: make configurable
    ground_truth_path = DATAPATH.parent.parent / "data" / "geoquat_ground_truth.json"
    predictions_path = DATAPATH / "output" / "predictions.json"
    temp_directory = DATAPATH / "_temp"

    with open(predictions_path, encoding="utf8") as file:
        predictions = json.load(file)

    evaluate(predictions, ground_truth_path, temp_directory, input_directory=None, draw_directory=None)
