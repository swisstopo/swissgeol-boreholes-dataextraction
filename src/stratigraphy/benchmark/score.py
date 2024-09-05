"""Evaluate the predictions against the ground truth."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import DatasetMetrics, DatasetMetricsCatalog, Metrics
from stratigraphy.util.draw import draw_predictions
from stratigraphy.util.predictions import FilePredictions
from stratigraphy.util.util import parse_text

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_layer_metrics(predictions: dict, number_of_truth_values: dict) -> DatasetMetrics:
    """Calculate F1, precision and recall for the layer predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        DatasetMetrics: the metrics for the layers
    """
    layer_metrics = DatasetMetrics()

    for filename, file_prediction in predictions.items():
        hits = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                hits += 1
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")
        layer_metrics.metrics[filename] = Metrics(
            tp=hits, fp=len(file_prediction.layers) - hits, fn=number_of_truth_values[filename] - hits
        )

    return layer_metrics


def get_depth_interval_metrics(predictions: dict) -> DatasetMetrics:
    """Calculate F1, precision and recall for the depth interval predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    Depth interval accuracy is not calculated for layers with incorrect material predictions.

    Args:
        predictions (dict): The predictions.

    Returns:
        DatasetMetrics: the metrics for the depth intervals
    """
    depth_interval_metrics = DatasetMetrics()

    for filename, file_prediction in predictions.items():
        depth_interval_hits = 0
        depth_interval_occurences = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                if layer.depth_interval_is_correct is not None:
                    depth_interval_occurences += 1
                if layer.depth_interval_is_correct:
                    depth_interval_hits += 1

        if depth_interval_occurences > 0:
            depth_interval_metrics.metrics[filename] = Metrics(
                tp=depth_interval_hits, fp=depth_interval_occurences - depth_interval_hits, fn=0
            )

    return depth_interval_metrics


def evaluate_borehole_extraction(
    predictions: dict[str, FilePredictions], number_of_truth_values: dict
) -> DatasetMetricsCatalog:
    """Evaluate the borehole extraction predictions.

    Args:
       predictions (dict): The FilePredictions objects.
       number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        DatasetMetricsCatalogue: A DatasetMetricsCatalogue that maps a metrics name to the corresponding DatasetMetrics
                                 object
    """
    all_metrics = evaluate_layer_extraction(predictions, number_of_truth_values)
    all_metrics.metrics["coordinates"] = get_metrics(predictions, "metadata_is_correct", "coordinates")
    all_metrics.metrics["elevation"] = get_metrics(predictions, "metadata_is_correct", "elevation")
    all_metrics.metrics["groundwater"] = get_metrics(predictions, "groundwater_is_correct", "groundwater")
    all_metrics.metrics["groundwater_depth"] = get_metrics(predictions, "groundwater_is_correct", "groundwater_depth")
    return all_metrics


def get_metrics(predictions: dict[str, FilePredictions], field_key: str, field_name: str) -> DatasetMetrics:
    """Get the metrics for a specific field in the predictions.

    Args:
        predictions (dict): The FilePredictions objects.
        field_key (str): The key to access the specific field in the prediction objects.
        field_name (str): The name of the field being evaluated.

    Returns:
        DatasetMetrics: The requested DatasetMetrics object.
    """
    dataset_metrics = DatasetMetrics()

    for file_name, file_prediction in predictions.items():
        dataset_metrics.metrics[file_name] = getattr(file_prediction, field_key)[field_name]

    return dataset_metrics


def evaluate_layer_extraction(predictions: dict, number_of_truth_values: dict) -> DatasetMetricsCatalog:
    """Calculate F1, precision and recall for the predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.
    The individual document metrics are returned as a DataFrame.

    Args:
        predictions (dict): The FilePredictions objects.
        number_of_truth_values (dict): The number of layer ground truth values per file.

    Returns:
        DatasetMetricsCatalogue: A dictionary that maps a metrics name to the corresponding DatasetMetrics object
    """
    all_metrics = DatasetMetricsCatalog()
    all_metrics.metrics["layer"] = get_layer_metrics(predictions, number_of_truth_values)
    all_metrics.metrics["depth_interval"] = get_depth_interval_metrics(predictions)

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
        all_metrics.metrics[f"{language}_layer"] = get_layer_metrics(
            language_predictions, language_number_of_truth_values
        )
        all_metrics.metrics[f"{language}_depth_interval"] = get_depth_interval_metrics(language_predictions)

    logging.info("Macro avg:")
    logging.info(
        f"F1: {all_metrics.metrics['layer'].macro_f1():.1%}, "
        f"precision: {all_metrics.metrics['layer'].macro_precision():.1%}, "
        f"recall: {all_metrics.metrics['layer'].macro_recall():.1%}, "
        f"depth_interval_accuracy: {all_metrics.metrics['depth_interval'].macro_precision():.1%}"
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

        metrics.document_level_metrics_df().to_csv(
            temp_directory / "document_level_metrics.csv", index_label="document_name"
        )  # mlflow.log_artifact expects a file

        metrics_dict = metrics.metrics_dict()
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
    ground_truth_path = DATAPATH.parent.parent / "data" / "zurich_ground_truth.json"
    predictions_path = DATAPATH / "output" / "predictions.json"
    temp_directory = DATAPATH / "_temp"

    with open(predictions_path, encoding="utf8") as file:
        predictions = json.load(file)

    evaluate(predictions, ground_truth_path, temp_directory, input_directory=None, draw_directory=None)
