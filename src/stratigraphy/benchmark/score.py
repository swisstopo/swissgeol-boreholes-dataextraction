"""Evaluate the predictions against the ground truth."""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from stratigraphy import DATAPATH
from stratigraphy.annotations.draw import draw_predictions
from stratigraphy.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics
from stratigraphy.util.predictions import OverallFilePredictions

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


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

    # TODO read BoreholeMetadataList from JSON file and pass to the evaluate method
    evaluate(predictions, ground_truth_path, temp_directory, input_directory=None, draw_directory=None)
