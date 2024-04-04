"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from pathlib import Path

import click
import fitz
from dotenv import load_dotenv

from stratigraphy import DATAPATH
from stratigraphy.benchmark.score import add_ground_truth_to_predictions, evaluate_matching
from stratigraphy.extract import process_page
from stratigraphy.line_detection import extract_lines, line_detection_params
from stratigraphy.util.draw import draw_predictions
from stratigraphy.util.plot_utils import plot_lines
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
    type=click.Path(exists=False, path_type=Path),
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

    \f
    Args:
        input_directory (Path): The directory containing the pdf files.
        ground_truth_path (Path): The path to the ground truth file json file.
        out_directory (Path): The directory to store the evaluation results.
        predictions_path (Path): The path to the predictions file.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.
        draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.
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
        file_iterator = [(input_directory.parent, None, [input_directory.name])]
    else:
        file_iterator = os.walk(input_directory)
    # process the individual pdf files
    predictions = {}
    for root, _dirs, files in file_iterator:
        for filename in files:
            if filename.endswith(".pdf"):
                in_path = os.path.join(root, filename)
                logger.info("Processing file: %s", in_path)
                predictions[filename] = {}

                with fitz.Document(in_path) as doc:
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        logger.info("Processing page %s", page_number)

                        geometric_lines = extract_lines(page, line_detection_params)
                        layer_predictions, depths_materials_column_pairs = process_page(
                            page, geometric_lines, **matching_params
                        )

                        predictions[filename][f"page_{page_number}"] = {
                            "layers": layer_predictions,
                            "depths_materials_column_pairs": depths_materials_column_pairs,
                        }
                        if draw_lines:  # could be changed to if draw_lines and mflow_tracking:
                            if not mlflow_tracking:
                                logger.warning(
                                    "MLFlow tracking is not enabled. MLFLow is required to store the images."
                                )
                            else:
                                img = plot_lines(
                                    page, geometric_lines, scale_factor=line_detection_params["pdf_scale_factor"]
                                )
                                mlflow.log_image(img, f"pages/{filename}_page_{page.number + 1}_lines.png")

    with open(predictions_path, "w") as file:
        file.write(json.dumps(predictions))

    # evaluate the predictions
    predictions, number_of_truth_values = add_ground_truth_to_predictions(predictions, ground_truth_path)

    if not skip_draw_predictions:
        draw_predictions(predictions, input_directory, out_directory)

    if number_of_truth_values:  # only evaluate if ground truth is available
        metrics, document_level_metrics = evaluate_matching(predictions, number_of_truth_values)
        document_level_metrics.to_csv(
            temp_directory / "document_level_metrics.csv"
        )  # mlflow.log_artifact expects a file

        if mlflow_tracking:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(temp_directory / "document_level_metrics.csv")
    else:
        logger.warning("Ground truth file not found. Skipping evaluation.")


if __name__ == "__main__":
    start_pipeline()
