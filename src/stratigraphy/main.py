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
from stratigraphy.annotations.draw import draw_predictions
from stratigraphy.annotations.plot_utils import plot_lines
from stratigraphy.benchmark.score import evaluate
from stratigraphy.extract import MaterialDescriptionRectWithSidebarExtractor
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInDocument
from stratigraphy.layer.duplicate_detection import remove_duplicate_layers
from stratigraphy.layer.layer import LayersInDocument
from stratigraphy.lines.line_detection import extract_lines, line_detection_params
from stratigraphy.metadata.metadata import BoreholeMetadata
from stratigraphy.text.extract_text import extract_text_lines
from stratigraphy.util.predictions import FilePredictions, OverallFilePredictions
from stratigraphy.util.util import flatten, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow
    import pygit2

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")


def common_options(f):
    """Decorator to add common options to both commands."""
    f = click.option(
        "-i",
        "--input-directory",
        required=True,
        type=click.Path(exists=True, path_type=Path),
        help="Path to the input directory, or path to a single pdf file.",
    )(f)
    f = click.option(
        "-g",
        "--ground-truth-path",
        type=click.Path(exists=True, path_type=Path),
        help="Path to the ground truth file (optional).",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output",
        help="Path to the output directory.",
    )(f)
    f = click.option(
        "-p",
        "--predictions-path",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output" / "predictions.json",
        help="Path to the predictions file.",
    )(f)
    f = click.option(
        "-m",
        "--metadata-path",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output" / "metadata.json",
        help="Path to the metadata file.",
    )(f)
    f = click.option(
        "-s",
        "--skip-draw-predictions",
        is_flag=True,
        default=False,
        help="Whether to skip drawing the predictions on pdf pages. Defaults to False.",
    )(f)
    f = click.option(
        "-l",
        "--draw-lines",
        is_flag=True,
        default=False,
        help="Whether to draw lines on pdf pages. Defaults to False.",
    )(f)
    return f


@click.command()
@common_options
@click.option(
    "-pa", "--part", type=click.Choice(["all", "metadata"]), default="all", help="The part of the pipeline to run."
)
def click_pipeline(
    input_directory: Path,
    ground_truth_path: Path | None,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    part: str = "all",
):
    """Run the boreholes data extraction pipeline."""
    start_pipeline(
        input_directory=input_directory,
        ground_truth_path=ground_truth_path,
        out_directory=out_directory,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        skip_draw_predictions=skip_draw_predictions,
        draw_lines=draw_lines,
        part=part,
    )


@click.command()
@common_options
def click_pipeline_metadata(
    input_directory: Path,
    ground_truth_path: Path | None,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
):
    """Run only the metadata part of the pipeline."""
    start_pipeline(
        input_directory=input_directory,
        ground_truth_path=ground_truth_path,
        out_directory=out_directory,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        skip_draw_predictions=skip_draw_predictions,
        draw_lines=draw_lines,
        part="metadata",
    )


def setup_mlflow_tracking(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path = None,
    predictions_path: Path = None,
    metadata_path: Path = None,
    experiment_name: str = "Boreholes data extraction",
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

    repo = pygit2.Repository(".")
    commit = repo[repo.head.target]
    mlflow.set_tag("git_branch", repo.head.shorthand)
    mlflow.set_tag("git_commit_message", commit.message)
    mlflow.set_tag("git_commit_sha", commit.id)


def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    part: str = "all",
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
        part (str, optional): The part of the pipeline to run. Defaults to "all".
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
        root = input_directory.parent
        files = [input_directory.name]
    else:
        root = input_directory
        _, _, files = next(os.walk(input_directory))

    # process the individual pdf files
    predictions = OverallFilePredictions()

    for filename in tqdm(files, desc="Processing files", unit="file"):
        if filename.endswith(".pdf"):
            in_path = os.path.join(root, filename)
            logger.info("Processing file: %s", in_path)

            with fitz.Document(in_path) as doc:
                # Extract metadata
                metadata = BoreholeMetadata.from_document(doc)

                # Save the predictions to the overall predictions object
                # Initialize common variables
                layers_in_document = LayersInDocument([], filename)
                bounding_boxes = []
                aggregated_groundwater_entries = []

                if part == "all":
                    # Extract the layers
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        logger.info("Processing page %s", page_number)

                        text_lines = extract_text_lines(page)
                        geometric_lines = extract_lines(page, line_detection_params)
                        process_page_results = MaterialDescriptionRectWithSidebarExtractor(
                            text_lines, geometric_lines, metadata.language, page_number, **matching_params
                        ).process_page()

                        # Extract the groundwater levels
                        for page_bounding_box in process_page_results.bounding_boxes:
                            material_description_bbox = page_bounding_box.material_description_bbox

                            groundwater_entries_near_bbox = GroundwaterInDocument.near_material_description(
                                document=doc,
                                page_number=page_number,
                                lines=text_lines,
                                material_description_bbox=material_description_bbox,
                                terrain_elevation=metadata.elevation,
                            )
                            ##avoid duplicate entries
                            for groundwater_entry in groundwater_entries_near_bbox:
                                if groundwater_entry not in aggregated_groundwater_entries:
                                    aggregated_groundwater_entries.append(groundwater_entry)

                        # TODO: Add remove duplicates here!
                        if page_index > 0:
                            layer_predictions = remove_duplicate_layers(
                                previous_page=doc[page_index - 1],
                                current_page=page,
                                previous_layers=layers_in_document,
                                current_layers=process_page_results.predictions,
                                img_template_probability_threshold=matching_params[
                                    "img_template_probability_threshold"
                                ],
                            )
                        else:
                            layer_predictions = process_page_results.predictions

                        layers_in_document.layers.extend(layer_predictions)
                        bounding_boxes.extend(process_page_results.bounding_boxes)

                        if draw_lines:  # could be changed to if draw_lines and mlflow_tracking:
                            if not mlflow_tracking:
                                logger.warning(
                                    "MLFlow tracking is not enabled. MLFLow is required to store the images."
                                )
                            else:
                                img = plot_lines(
                                    page, geometric_lines, scale_factor=line_detection_params["pdf_scale_factor"]
                                )
                                mlflow.log_image(img, f"pages/{filename}_page_{page.number + 1}_lines.png")

                # Create a document-level groundwater entry
                groundwater_entries = GroundwaterInDocument(
                    filename=filename,
                    groundwater=aggregated_groundwater_entries,
                )

                # Add file predictions
                predictions.add_file_predictions(
                    FilePredictions(
                        file_name=filename,
                        metadata=metadata,
                        groundwater=groundwater_entries,
                        layers_in_document=layers_in_document,
                        bounding_boxes=bounding_boxes,
                    )
                )

    logger.info("Metadata written to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(predictions.get_metadata_as_dict(), file, ensure_ascii=False)

    if part == "all":
        logger.info("Writing predictions to JSON file %s", predictions_path)
        with open(predictions_path, "w", encoding="utf8") as file:
            json.dump(predictions.to_json(), file, ensure_ascii=False)

    document_level_metadata_metrics = evaluate(
        predictions=predictions, ground_truth_path=ground_truth_path, temp_directory=temp_directory
    )

    if input_directory and draw_directory:
        draw_predictions(predictions, input_directory, draw_directory, document_level_metadata_metrics)


if __name__ == "__main__":
    click_pipeline()
