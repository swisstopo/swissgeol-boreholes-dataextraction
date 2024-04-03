"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
import shutil
from pathlib import Path

import click
from dotenv import load_dotenv

from stratigraphy import DATAPATH
from stratigraphy.benchmark.score import evaluate_matching
from stratigraphy.extract import perform_matching
from stratigraphy.line_detection import draw_lines_on_pdfs, line_detection_params
from stratigraphy.util.util import flatten, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")


@click.command()
@click.option(
    "-i",
    "--input_directory",
    type=click.Path(exists=True, path_type=Path),
    default=DATAPATH / "Benchmark",
    help="Path to the input directory.",
)
@click.option(
    "-g",
    "--ground_truth_path",
    type=click.Path(exists=True, path_type=Path),
    default=DATAPATH / "Benchmark" / "ground_truth.json",
    help="Path to the ground truth file.",
)
@click.option(
    "-o",
    "--out_directory",
    type=click.Path(path_type=Path),
    default=DATAPATH / "Benchmark" / "evaluation",
    help="Path to the output directory.",
)
@click.option(
    "-p",
    "--predictions_path",
    type=click.Path(path_type=Path),
    default=DATAPATH / "Benchmark" / "extract" / "predictions.json",
    help="Path to the predictions file.",
)
@click.option(
    "-s",
    "--skip-draw-predictions",
    is_flag=True,
    default=False,
    help="Whether to skip drawing the predictions on pdf pages. Defaults to False.",
)
@click.option(
    "-l", "--draw-lines", is_flag=True, default=False, help="Whether to draw lines on pdf pages. Defaults to False."
)
def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
):
    """Run the boreholes data extraction pipeline.

    The pipeline will extract material description of all found layers and assign them to the corresponding
    depth intervals. The input directory should contain pdf files with boreholes data. The algorithm can deal
    with borehole profiles of multiple pages.

    Args:\n
        input_directory (Path): The directory containing the pdf files.\n
        ground_truth_path (Path): The path to the ground truth file json file.\n
        out_directory (Path): The directory to store the evaluation results.\n
        predictions_path (Path): The path to the predictions file.\n
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.\n
        draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.\n
    """  # noqa: D301
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()
        mlflow.log_params(flatten(line_detection_params))
        mlflow.log_params(flatten(matching_params))

    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging

    # check if directories exist and create them when neccessary
    out_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    # if a file is specified instead of an input directory, copy the file to a temporary directory and work with that.
    if input_directory.is_file():
        if (temp_directory / "single_file").is_dir():
            shutil.rmtree(temp_directory / "single_file")

        Path.mkdir(temp_directory / "single_file")
        shutil.copy(input_directory, temp_directory / "single_file")
        input_directory = temp_directory / "single_file"

    # run the matching pipeline and save the result
    predictions = perform_matching(input_directory, **matching_params)
    with open(predictions_path, "w") as file:
        file.write(json.dumps(predictions))

    # evaluate the predictions
    metrics, document_level_metrics = evaluate_matching(
        predictions_path, ground_truth_path, input_directory, out_directory, skip_draw_predictions
    )
    document_level_metrics.to_csv(temp_directory / "document_level_metrics.csv")  # mlflow.log_artifact expects a file

    if mlflow_tracking:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(temp_directory / "document_level_metrics.csv")

    if draw_lines:
        logger.info("Drawing lines on pdf pages.")
        draw_lines_on_pdfs(input_directory, line_detection_params=line_detection_params)


if __name__ == "__main__":
    start_pipeline()
