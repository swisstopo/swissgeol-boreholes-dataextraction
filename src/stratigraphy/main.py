"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from pathlib import Path

import click
import fitz
from dotenv import load_dotenv

from stratigraphy import DATAPATH
from stratigraphy.benchmark.score import create_predictions_objects, evaluate_borehole_extraction
from stratigraphy.extract import process_page
from stratigraphy.line_detection import extract_lines, line_detection_params
from stratigraphy.util.coordinate_extraction import CoordinateExtractor
from stratigraphy.util.draw import draw_predictions
from stratigraphy.util.duplicate_detection import remove_duplicate_layers
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.language_detection import detect_language_of_document
from stratigraphy.util.plot_utils import plot_lines
from stratigraphy.util.util import flatten, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")


@click.command()
@click.option(
    "-i",
    "--input-directory",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input directory, or path to a single pdf file.",
)
@click.option(
    "-g",
    "--ground-truth-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ground truth file (optional).",
)
@click.option(
    "-o",
    "--out-directory",
    type=click.Path(path_type=Path),
    default=DATAPATH / "output",
    help="Path to the output directory.",
)
@click.option(
    "-p",
    "--predictions-path",
    type=click.Path(path_type=Path),
    default=DATAPATH / "output" / "predictions.json",
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
def click_pipeline(
    input_directory: Path,
    ground_truth_path: Path | None,
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
         input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
         ground_truth_path (Path | None): The path to the ground truth file json file.
         out_directory (Path): The directory to store the evaluation results.
         predictions_path (Path): The path to the predictions file.
         skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.
         draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.
    """  # noqa: D301
    start_pipeline(
        input_directory=input_directory,
        ground_truth_path=ground_truth_path,
        out_directory=out_directory,
        predictions_path=predictions_path,
        skip_draw_predictions=skip_draw_predictions,
        draw_lines=draw_lines,
    )


def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
) -> list[dict]:
    """Run the boreholes data extraction pipeline.

    The pipeline will extract material description of all found layers and assign them to the corresponding
    depth intervals. The input directory should contain pdf files with boreholes data. The algorithm can deal
    with borehole profiles of multiple pages.

    Note: This function is used to be called from the label-studio backend, whereas the click_pipeline function
    is called from the CLI.

    Args:
        input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
        ground_truth_path (Path | None): The path to the ground truth file json file.
        out_directory (Path): The directory to store the evaluation results.
        predictions_path (Path): The path to the predictions file.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.
        draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.

    Returns:
        list[dict]: The predictions of the pipeline.
    """  # noqa: D301
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()
        mlflow.set_tag("input_directory", str(input_directory))
        mlflow.set_tag("ground_truth_path", str(ground_truth_path))
        mlflow.set_tag("out_directory", str(out_directory))
        mlflow.set_tag("predictions_path", str(predictions_path))
        mlflow.log_params(flatten(line_detection_params))
        mlflow.log_params(flatten(matching_params))

    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging

    # check if directories exist and create them when necessary
    draw_directory = out_directory / "draw"
    draw_directory.mkdir(parents=True, exist_ok=True)
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
                    language = detect_language_of_document(
                        doc, matching_params["default_language"], matching_params["material_description"].keys()
                    )
                    predictions[filename]["language"] = language
                    coordinate_extractor = CoordinateExtractor(doc)
                    coordinates = coordinate_extractor.extract_coordinates()
                    if coordinates:
                        predictions[filename]["metadata"] = {"coordinates": coordinates.to_json()}
                    else:
                        predictions[filename]["metadata"] = {"coordinates": None}
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        logger.info("Processing page %s", page_number)

                        text_lines = extract_text_lines(page)
                        geometric_lines = extract_lines(page, line_detection_params)
                        layer_predictions, depths_materials_column_pairs = process_page(
                            text_lines, geometric_lines, language, **matching_params
                        )
                        # Add remove duplicates here!
                        if page_index > 0:
                            layer_predictions = remove_duplicate_layers(
                                doc[page_index - 1],
                                page,
                                predictions[filename][f"page_{page_number - 1}"]["layers"],
                                layer_predictions,
                                matching_params["img_template_probability_threshold"],
                            )
                        predictions[filename][f"page_{page_number}"] = {
                            "layers": layer_predictions,
                            "depths_materials_column_pairs": depths_materials_column_pairs,
                            "page_height": page.rect.height,
                            "page_width": page.rect.width,
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

    logger.info("Writing predictions to JSON file %s", predictions_path)
    with open(predictions_path, "w") as file:
        file.write(json.dumps(predictions))

    # evaluate the predictions; if file doesnt exist, the predictions are not changed.
    predictions, number_of_truth_values = create_predictions_objects(predictions, ground_truth_path)

    if not skip_draw_predictions:
        draw_predictions(predictions, input_directory, draw_directory)

    if number_of_truth_values:  # only evaluate if ground truth is available
        metrics, document_level_metrics = evaluate_borehole_extraction(predictions, number_of_truth_values)
        document_level_metrics.to_csv(
            temp_directory / "document_level_metrics.csv"
        )  # mlflow.log_artifact expects a file

        if mlflow_tracking:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(temp_directory / "document_level_metrics.csv")
    else:
        logger.warning("Ground truth file not found. Skipping evaluation.")

    return predictions


if __name__ == "__main__":
    click_pipeline()
