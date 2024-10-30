"""Evaluate the predictions against the ground truth."""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.annotations.draw import draw_predictions
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics
from stratigraphy.util.predictions import OverallFilePredictions

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def create_predictions_objects(
    predictions: OverallFilePredictions,
    ground_truth_path: Path | None,
) -> tuple[OverallFilePredictions, dict]:
    """Create predictions objects from the predictions and evaluate them against the ground truth.

    Args:
        predictions (OverallFilePredictions): The predictions objects.
        ground_truth_path (Path | None): The path to the ground truth file.

    Returns:
        tuple[OverallFilePredictions, dict]: The predictions objects and the number of ground truth values per
                                                 file.
    """
    if ground_truth_path and ground_truth_path.exists():  # for inference no ground truth is available
        ground_truth = GroundTruth(ground_truth_path)
        ground_truth_is_present = True
    else:
        logging.warning("Ground truth file not found.")
        ground_truth_is_present = False
        return predictions, {}

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
) -> None:
    """Computes all the metrics, logs them, and creates corresponding MLFlow artifacts (when enabled).

    Args:
        predictions (OverallFilePredictions): The predictions objects.
        ground_truth_path (Path): The path to the ground truth file.
        temp_directory (Path): The path to the temporary directory.
        input_directory (Path | None): The path to the input directory.
        draw_directory (Path | None): The path to the draw directory.

    Returns:
        None
    """
    #############################
    # Evaluate the borehole extraction metadata
    #############################
    metadata_metrics_list = predictions.evaluate_metadata_extraction(ground_truth_path)
    metadata_metrics: BoreholeMetadataMetrics = metadata_metrics_list.get_cumulated_metrics()
    document_level_metadata_metrics: pd.DataFrame = metadata_metrics_list.get_document_level_metrics()
    document_level_metadata_metrics.to_csv(
        temp_directory / "document_level_metadata_metrics.csv", index_label="document_name"
    )  # mlflow.log_artifact expects a file

    # print the metrics
    logger.info("Metadata Performance metrics:")
    logger.info(metadata_metrics.to_json())

    if mlflow_tracking:
        import mlflow

        mlflow.log_metrics(metadata_metrics.to_json())
        mlflow.log_artifact(temp_directory / "document_level_metadata_metrics.csv")

    #############################
    # Evaluate the borehole extraction
    #############################
    metrics = predictions.evaluate_borehole_extraction(ground_truth_path)

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

    # Load the predictions
    try:
        with open(args.predictions_path, encoding="utf8") as file:
            predictions = json.load(file)
    except FileNotFoundError:
        logger.error("Predictions file not found: %s", args.predictions_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from predictions file: %s", e)
        return

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
