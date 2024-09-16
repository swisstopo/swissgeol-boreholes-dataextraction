"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from pathlib import Path

import click
import fitz
from dotenv import load_dotenv
from tqdm import tqdm

from stratigraphy import DATAPATH
from stratigraphy.annotations.plot_utils import plot_lines
from stratigraphy.benchmark.score import (
    evaluate,
    evaluate_metadata_extraction,
)
from stratigraphy.extract import process_page
from stratigraphy.groundwater.groundwater_extraction import GroundwaterLevelExtractor
from stratigraphy.layer.duplicate_detection import remove_duplicate_layers
from stratigraphy.lines.line_detection import extract_lines, line_detection_params
from stratigraphy.metadata.metadata import BoreholeMetadata, BoreholeMetadataList
from stratigraphy.text.extract_text import extract_text_lines
from stratigraphy.util.util import flatten, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

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
    "-m",
    "--metadata-path",
    type=click.Path(path_type=Path),
    default=DATAPATH / "output" / "metadata.json",
    help="Path to the metadata file.",
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
    metadata_path: Path,
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
        metadata_path=metadata_path,
        skip_draw_predictions=skip_draw_predictions,
        draw_lines=draw_lines,
    )


# Add command to extract metadata
@click.command()
@click.option(
    "-i",
    "--input-directory",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input directory, or path to a single pdf file.",
)
@click.option(
    "-m",
    "--metadata-path",
    type=click.Path(path_type=Path),
    default=DATAPATH / "output" / "metadata.json",
    help="Path to the metadata file.",
)
@click.option(
    "-g",
    "--ground-truth-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ground truth file.",
)
def click_metadata_pipeline(input_directory: Path, metadata_path: Path, ground_truth_path: Path):
    """Run the metadata extraction pipeline.

    The pipeline will extract metadata from boreholes data. The input directory should contain pdf files
    with boreholes data. The algorithm can deal with borehole profiles of multiple pages.

    Args:
        input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
        metadata_path (Path): The path to the metadata file.
        ground_truth_path (Path): The path to the ground truth file.
    """
    start_metadata_pipeline(input_directory, metadata_path, ground_truth_path)


def setup_mlflow_tracking(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path = None,
    predictions_path: Path = None,
    metadata_path: Path = None,
    experiment_name: str = "Boreholes Stratigraphy",
):
    """Set up MLFlow tracking."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("input_directory", str(input_directory))
    mlflow.set_tag("ground_truth_path", str(ground_truth_path))
    if out_directory:
        mlflow.set_tag("out_directory", str(out_directory))
    if predictions_path:
        mlflow.set_tag("predictions_path", str(predictions_path))
    if metadata_path:
        mlflow.set_tag("metadata_path", str(metadata_path))
    mlflow.log_params(flatten(line_detection_params))
    mlflow.log_params(flatten(matching_params))


def start_metadata_pipeline(input_directory: Path, metadata_path: Path, ground_truth_path: Path) -> list[dict]:
    """Run the metadata extraction pipeline.

    The pipeline will extract metadata from boreholes data. The input directory should contain pdf files
    with boreholes data. The algorithm can deal with borehole profiles of multiple pages.

    Args:
        input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
        metadata_path (Path): The path to the metadata file.
        ground_truth_path (Path): The path to the ground truth file.

    Returns:
        list[dict]: The metadata of the pipeline.
    """  # noqa: D301
    if mlflow_tracking:
        setup_mlflow_tracking(
            input_directory=input_directory,
            ground_truth_path=ground_truth_path,
            out_directory=None,
            predictions_path=None,
            metadata_path=metadata_path,
        )

    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging

    # check if directories exist and create them when necessary
    temp_directory.mkdir(parents=True, exist_ok=True)

    # if a file is specified instead of an input directory, copy the file to a temporary directory and work with that.
    if input_directory.is_file():
        file_iterator = [(input_directory.parent, None, [input_directory.name])]
    else:
        file_iterator = os.walk(input_directory)
    # process the individual pdf files
    metadata_per_file = BoreholeMetadataList()

    for root, _dirs, files in file_iterator:
        for filename in tqdm(files, desc="Processing files", unit="file"):
            if filename.endswith(".pdf"):
                in_path = os.path.join(root, filename)
                logger.info("Processing file: %s", in_path)

                with fitz.Document(in_path) as doc:
                    # Extract metadata
                    metadata = BoreholeMetadata(doc)

                    # Add metadata to the metadata list
                    metadata_per_file.metadata_per_file.append(metadata)

    logger.info("Metadata written to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(metadata_per_file.to_json(), file, ensure_ascii=False)

    # Evaluate the metadata
    metadata_metrics_list = evaluate_metadata_extraction(metadata_per_file, ground_truth_path)
    metadata_metrics = metadata_metrics_list.get_cumulated_metrics()
    document_level_metrics = metadata_metrics_list.get_document_level_metrics()

    document_level_metrics.to_csv(
        temp_directory / "document_level_metadata_metrics.csv"
    )  # mlflow.log_artifact expects a file

    # print the metrics
    logger.info("Performance metrics:")
    logger.info(metadata_metrics)

    if mlflow_tracking:
        mlflow.log_metrics(metadata_metrics)
        mlflow.log_artifact(temp_directory / "document_level_metadata_metrics.csv")
    else:
        logger.warning("Ground truth file not found. Skipping evaluation.")


def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
):
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
        metadata_path (Path): The path to the metadata file.
    """  # noqa: D301
    if mlflow_tracking:
        setup_mlflow_tracking(input_directory, ground_truth_path, out_directory, predictions_path, metadata_path)

    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging
    temp_directory.mkdir(parents=True, exist_ok=True)

    if skip_draw_predictions:
        draw_directory = None
    else:
        # check if directories exist and create them when necessary
        draw_directory = out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

    # if a file is specified instead of an input directory, copy the file to a temporary directory and work with that.
    if input_directory.is_file():
        file_iterator = [(input_directory.parent, None, [input_directory.name])]
    else:
        file_iterator = os.walk(input_directory)
    # process the individual pdf files
    predictions = {}
    metadata_per_file = BoreholeMetadataList()

    for root, _dirs, files in file_iterator:
        for filename in tqdm(files, desc="Processing files", unit="file"):
            if filename.endswith(".pdf"):
                in_path = os.path.join(root, filename)
                logger.info("Processing file: %s", in_path)
                predictions[filename] = {}

                with fitz.Document(in_path) as doc:
                    # Extract metadata
                    metadata = BoreholeMetadata(doc)

                    # Extract the groundwater levels
                    groundwater_extractor = GroundwaterLevelExtractor(document=doc)
                    groundwater = groundwater_extractor.extract_groundwater()
                    if groundwater:
                        predictions[filename]["groundwater"] = [
                            groundwater_entry.to_json() for groundwater_entry in groundwater
                        ]
                    else:
                        predictions[filename]["groundwater"] = None

                    layer_predictions_list = []
                    depths_materials_column_pairs_list = []
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        logger.info("Processing page %s", page_number)

                        text_lines = extract_text_lines(page)
                        geometric_lines = extract_lines(page, line_detection_params)
                        layer_predictions, depths_materials_column_pairs = process_page(
                            text_lines, geometric_lines, metadata.language, page_number, **matching_params
                        )

                        # TODO: Add remove duplicates here!
                        if page_index > 0:
                            layer_predictions = remove_duplicate_layers(
                                doc[page_index - 1],
                                page,
                                layer_predictions_list,
                                layer_predictions,
                                matching_params["img_template_probability_threshold"],
                            )

                        layer_predictions_list.extend(layer_predictions)
                        depths_materials_column_pairs_list.extend(depths_materials_column_pairs)

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

                    predictions[filename]["layers"] = layer_predictions_list
                    predictions[filename]["depths_materials_column_pairs"] = depths_materials_column_pairs_list
                    predictions[filename]["page_dimensions"] = metadata.page_dimensions

                    # Add metadata to the metadata list
                    metadata_per_file.metadata_per_file.append(metadata)

                    assert len(metadata.page_dimensions) == doc.page_count, "Page count mismatch."

    logger.info("Writing predictions to JSON file %s", predictions_path)
    with open(predictions_path, "w", encoding="utf8") as file:
        json.dump(predictions, file, ensure_ascii=False)

    logger.info("Metadata written to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(metadata_per_file.to_json(), file, ensure_ascii=False)

    evaluate(
        predictions=predictions,
        metadata_per_file=metadata_per_file,
        ground_truth_path=ground_truth_path,
        temp_directory=temp_directory,
        input_directory=input_directory,
        draw_directory=draw_directory,
    )


if __name__ == "__main__":
    click_pipeline()
    # click_metadata_pipeline()
