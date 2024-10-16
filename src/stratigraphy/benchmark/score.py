"""Evaluate the predictions against the ground truth."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.annotations.draw import draw_predictions
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import DatasetMetrics, DatasetMetricsCatalog, Metrics
from stratigraphy.util.predictions import OverallFilePredictions
from stratigraphy.util.util import parse_text

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_layer_metrics(predictions: OverallFilePredictions, number_of_truth_values: dict) -> DatasetMetrics:
    """Calculate F1, precision and recall for the layer predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    Args:
        predictions (dict): The predictions.
        number_of_truth_values (dict): The number of ground truth values per file.

    Returns:
        DatasetMetrics: the metrics for the layers
    """
    layer_metrics = DatasetMetrics()

    for file_prediction in predictions.file_predictions_list:
        hits = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                hits += 1
            if parse_text(layer.material_description.text) == "":
                logger.warning("Empty string found in predictions")
        layer_metrics.metrics[file_prediction.file_name] = Metrics(
            tp=hits,
            fp=len(file_prediction.layers) - hits,
            fn=number_of_truth_values.get(file_prediction.file_name, 0) - hits,
        )

    return layer_metrics


def get_depth_interval_metrics(predictions: OverallFilePredictions) -> DatasetMetrics:
    """Calculate F1, precision and recall for the depth interval predictions.

    Calculate F1, precision and recall for the individual documents as well as overall.

    Depth interval accuracy is not calculated for layers with incorrect material predictions.

    Args:
        predictions (dict): The predictions.

    Returns:
        DatasetMetrics: the metrics for the depth intervals
    """
    depth_interval_metrics = DatasetMetrics()

    for file_prediction in predictions.file_predictions_list:
        depth_interval_hits = 0
        depth_interval_occurrences = 0
        for layer in file_prediction.layers:
            if layer.material_is_correct:
                if layer.depth_interval_is_correct is not None:
                    depth_interval_occurrences += 1
                if layer.depth_interval_is_correct:
                    depth_interval_hits += 1

        if depth_interval_occurrences > 0:
            depth_interval_metrics.metrics[file_prediction.file_name] = Metrics(
                tp=depth_interval_hits, fp=depth_interval_occurrences - depth_interval_hits, fn=0
            )

    return depth_interval_metrics


def evaluate_borehole_extraction(
    predictions: OverallFilePredictions, number_of_truth_values: dict
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
    all_metrics.metrics["groundwater"] = get_metrics(predictions, "groundwater_is_correct", "groundwater")
    all_metrics.metrics["groundwater_depth"] = get_metrics(predictions, "groundwater_is_correct", "groundwater_depth")
    return all_metrics


def get_metrics(predictions: OverallFilePredictions, field_key: str, field_name: str) -> DatasetMetrics:
    """Get the metrics for a specific field in the predictions.

    Args:
        predictions (dict): The FilePredictions objects.
        field_key (str): The key to access the specific field in the prediction objects.
        field_name (str): The name of the field being evaluated.

    Returns:
        DatasetMetrics: The requested DatasetMetrics object.
    """
    dataset_metrics = DatasetMetrics()

    for file_prediction in predictions.file_predictions_list:
        attribute = getattr(file_prediction, field_key, None)
        if attribute and field_name in attribute:
            dataset_metrics.metrics[file_prediction.file_name] = attribute[field_name]
        else:
            logger.warning(
                "Missing attribute '%s' or key '%s' in file '%s'", field_key, field_name, file_prediction.file_name
            )

    return dataset_metrics


def evaluate_layer_extraction(
    predictions: OverallFilePredictions, number_of_truth_values: dict
) -> DatasetMetricsCatalog:
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
    languages = set(fp.metadata.language for fp in predictions.file_predictions_list)
    predictions_by_language = {language: OverallFilePredictions() for language in languages}

    for file_predictions in predictions.file_predictions_list:
        language = file_predictions.metadata.language
        if language in predictions_by_language:
            predictions_by_language[language].add_file_predictions(file_predictions)

    for language, language_predictions in predictions_by_language.items():
        language_number_of_truth_values = {
            prediction.file_name: number_of_truth_values[prediction.file_name]
            for prediction in language_predictions.file_predictions_list
        }
        all_metrics.metrics[f"{language}_layer"] = get_layer_metrics(
            language_predictions, language_number_of_truth_values
        )
        all_metrics.metrics[f"{language}_depth_interval"] = get_depth_interval_metrics(language_predictions)

    logging.info("Macro avg:")
    logging.info(
        "F1: %.1f%%, precision: %.1f%%, recall: %.1f%%, depth_interval_accuracy: %.1f%%",
        all_metrics.metrics["layer"].macro_f1() * 100,
        all_metrics.metrics["layer"].macro_precision() * 100,
        all_metrics.metrics["layer"].macro_recall() * 100,
        all_metrics.metrics["depth_interval"].macro_precision() * 100,
    )

    return all_metrics


def create_predictions_objects(
    predictions: OverallFilePredictions,
    ground_truth_path: Path | None,
) -> tuple[OverallFilePredictions, dict]:
    """Create predictions objects from the predictions and evaluate them against the ground truth.

    Args:
        predictions (dict): The predictions from the predictions.json file.
        ground_truth_path (Path | None): The path to the ground truth file.

    Returns:
        tuple[OverallFilePredictions, dict]: The predictions objects and the number of ground truth values per
                                                 file.
    """
    if ground_truth_path and os.path.exists(ground_truth_path):  # for inference no ground truth is available
        ground_truth = GroundTruth(ground_truth_path)
        ground_truth_is_present = True
    else:
        logging.warning("Ground truth file not found.")
        ground_truth_is_present = False

    number_of_truth_values = {}
    for file_predictions in predictions.file_predictions_list:
        if ground_truth_is_present:
            ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
            if ground_truth_for_file:
                file_predictions.evaluate(ground_truth_for_file)
                number_of_truth_values[file_predictions.file_name] = len(ground_truth_for_file["layers"])

    return predictions, number_of_truth_values


def evaluate(
    predictions: OverallFilePredictions,
    ground_truth_path: Path,
    temp_directory: Path,
    input_directory: Path | None,
    draw_directory: Path | None,
):
    """Computes all the metrics, logs them, and creates corresponding MLFlow artifacts (when enabled)."""
    #############################
    # Evaluate the borehole extraction metadata
    #############################
    metadata_metrics_list = predictions.evaluate_metadata_extraction(ground_truth_path)
    metadata_metrics = metadata_metrics_list.get_cumulated_metrics()
    document_level_metadata_metrics = metadata_metrics_list.get_document_level_metrics()
    document_level_metadata_metrics.to_csv(
        temp_directory / "document_level_metadata_metrics.csv", index_label="document_name"
    )  # mlflow.log_artifact expects a file

    # print the metrics
    logger.info("Metadata Performance metrics:")
    logger.info(metadata_metrics)

    if mlflow_tracking:
        import mlflow

        mlflow.log_metrics(metadata_metrics)
        mlflow.log_artifact(temp_directory / "document_level_metadata_metrics.csv")

    #############################
    # Evaluate the borehole extraction
    #############################
    if predictions:
        predictions, number_of_truth_values = create_predictions_objects(predictions, ground_truth_path)

        if number_of_truth_values:  # only evaluate if ground truth is available
            metrics = evaluate_borehole_extraction(predictions, number_of_truth_values)

            metrics.document_level_metrics_df().to_csv(
                temp_directory / "document_level_metrics.csv", index_label="document_name"
            )  # mlflow.log_artifact expects a file
            metrics_dict = metrics.metrics_dict()

            # Format the metrics dictionary to limit to three digits
            formatted_metrics = {k: f"{v:.3f}" for k, v in metrics_dict.items()}
            logger.info("Performance metrics: %s", formatted_metrics)

            if mlflow_tracking:
                mlflow.log_metrics(metrics_dict)
                mlflow.log_artifact(temp_directory / "document_level_metrics.csv")

        else:
            logger.warning("Ground truth file not found. Skipping evaluation.")

        #############################
        # Draw the prediction
        #############################
        if input_directory and draw_directory:
            draw_predictions(predictions, input_directory, draw_directory, document_level_metadata_metrics)


def main():
    """Main function to evaluate the predictions against the ground truth."""
    args = parse_cli()

    # setup mlflow tracking; should be started before any other code
    # such that tracking is enabled in other parts of the code.
    # This does not create any scores, but will log all the created images to mlflow.
    if args.mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()

    with open(args.predictions_path, encoding="utf8") as file:
        predictions = json.load(file)

    predictions = OverallFilePredictions.from_json(predictions)

    # Customize these as needed
    evaluate(predictions, args.ground_truth_path, args.temp_directory, input_directory=None, draw_directory=None)


def parse_cli() -> argparse.Namespace:
    """Parse the command line arguments and pass them to the main function."""
    parser = argparse.ArgumentParser(description="Borehole Stratigraphy Evaluation Script")

    # Add arguments with defaults
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=DATAPATH.parent / "data" / "zurich_ground_truth.json",
        help="Path to the ground truth JSON file (default: '../data/zurich_ground_truth.json').",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DATAPATH / "output" / "predictions.json",
        help="Path to the predictions JSON file (default: './output/predictions.json').",
    )
    parser.add_argument(
        "--temp-directory",
        type=Path,
        default=DATAPATH / "_temp",
        help="Directory for storing temporary data (default: './_temp').",
    )
    parser.add_argument(
        "--no-mlflow-tracking",
        action="store_false",
        dest="mlflow_tracking",
        help="Disable MLflow tracking (enabled by default).",
    )

    # Parse arguments and pass to main
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and pass to main
    main()
