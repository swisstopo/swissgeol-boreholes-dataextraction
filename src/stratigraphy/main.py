"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from stratigraphy import DATAPATH
from stratigraphy.benchmark.score import evaluate_matching
from stratigraphy.extract import perform_matching
from stratigraphy.line_detection import line_detection_params
from stratigraphy.util.util import flatten, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")


@click.command()
@click.option(
    "--input_directory",
    type=click.Path(exists=True, path_type=Path),
    default=DATAPATH / "Benchmark",
    help="Path to the input directory.",
)
@click.option(
    "--ground_truth_path",
    type=click.Path(exists=True, path_type=Path),
    default=DATAPATH / "Benchmark" / "ground_truth.json",
    help="Path to the ground truth file.",
)
@click.option(
    "--out_directory",
    type=click.Path(path_type=Path),
    default=DATAPATH / "Benchmark" / "evaluation",
    help="Path to the output directory.",
)
@click.option(
    "--predictions_path",
    type=click.Path(path_type=Path),
    default=DATAPATH / "Benchmark" / "extract" / "predictions.json",
    help="Path to the predictions file.",
)
def start_pipeline(input_directory: Path, ground_truth_path: Path, out_directory: Path, predictions_path: Path):
    """Description.

    Args:
        input_directory (Path): _description_
        ground_truth_path (Path): _description_
        out_directory (Path): _description_
        predictions_path (Path): _description_
    """
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()
        mlflow.log_params(flatten(line_detection_params))
        mlflow.log_params(flatten(matching_params))

    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging
    # check if directories exist and create them when neccessary
    # check if directories exist and create them when neccessary
    out_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    # run the matching pipeline and save the result
    predictions = perform_matching(input_directory, **matching_params)
    with open(predictions_path, "w") as file:
        file.write(json.dumps(predictions))

    # evaluate the predictions
    metrics, document_level_metrics = evaluate_matching(
        predictions_path, ground_truth_path, input_directory, out_directory
    )
    document_level_metrics.to_csv(temp_directory / "document_level_metrics.csv")  # mlflow.log_artifact expects a file

    if mlflow_tracking:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(temp_directory / "document_level_metrics.csv")


if __name__ == "__main__":
    start_pipeline()
