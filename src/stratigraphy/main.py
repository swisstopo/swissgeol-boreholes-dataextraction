"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from collections import defaultdict
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
from stratigraphy.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwaterLevelExtractor,
    GroundwatersInBorehole,
)
from stratigraphy.layer.duplicate_detection import remove_duplicate_layers
from stratigraphy.layer.layer import LayersInDocument
from stratigraphy.lines.line_detection import extract_lines, line_detection_params
from stratigraphy.metadata.metadata import FileMetadata, MetadataInDocument
from stratigraphy.text.extract_text import extract_text_lines
from stratigraphy.util.predictions import (
    BoreholeListBuilder,
    BoreholePredictions,
    FilePredictions,
    CsvPredictions,
    OverallFilePredictions,
)
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
    f = click.option(
        "-c",
        "--csv",
        is_flag=True,
        default=False,
        help="Whether to generate CSV output. Defaults to False.",
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
    csv: bool = False,
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
        csv=csv,
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
    csv: bool = False,
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
        if not filename.endswith(".pdf"):
            continue

        in_path = os.path.join(root, filename)
        logger.info("Processing file: %s", in_path)

        with fitz.Document(in_path) as doc:
            # Extract metadata
            file_metadata = FileMetadata.from_document(doc)
            metadata = MetadataInDocument.from_document(doc)

            # Save the predictions to the overall predictions object
            # Initialize common variables
            layers_in_document = LayersInDocument([], filename)
            bounding_boxes = []
            aggregated_groundwater_entries = defaultdict(list)

            if part != "all":
                continue
            # Extract the layers
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                logger.info("Processing page %s", page_number)

                text_lines = extract_text_lines(page)
                geometric_lines = extract_lines(page, line_detection_params)
                process_page_results = MaterialDescriptionRectWithSidebarExtractor(
                    text_lines, geometric_lines, file_metadata.language, page_number, **matching_params
                ).process_page()

                # Extract the groundwater levels
                for borehole_index, page_bounding_box in enumerate(process_page_results.bounding_boxes):
                    material_description_bbox = page_bounding_box.material_description_bbox

                    groundwater_entries_near_bbox = GroundwaterLevelExtractor.near_material_description(
                        document=doc,
                        page_number=page_number,
                        lines=text_lines,
                        material_description_bbox=material_description_bbox,
                        terrain_elevations=metadata.elevations if metadata.elevations else None,
                    )
                    ##avoid duplicate entries
                    for groundwater_entry in groundwater_entries_near_bbox:
                        if groundwater_entry not in [
                            already_seen
                            for borehole_gw in aggregated_groundwater_entries.values()
                            for already_seen in borehole_gw
                        ]:
                            aggregated_groundwater_entries[borehole_index].append(groundwater_entry)

                # TODO: Add remove duplicates here!
                if page_index > 0:
                    layer_predictions = remove_duplicate_layers(
                        previous_page=doc[page_index - 1],
                        current_page=page,
                        previous_layers=layers_in_document,
                        current_layers=process_page_results.predictions,
                        img_template_probability_threshold=matching_params["img_template_probability_threshold"],
                    )
                else:
                    layer_predictions = process_page_results.predictions

                layers_in_document.assign_layers_to_boreholes(layer_predictions)
                if page_number == 1:
                    bounding_boxes = [[bbox] for bbox in process_page_results.bounding_boxes]
                else:
                    # use assumption, if there is multiple pages, there is only one borehole (so one bbox too)
                    if bounding_boxes:
                        bounding_boxes[0].extend(process_page_results.bounding_boxes)
                    else:
                        bounding_boxes = [[bbox] for bbox in process_page_results.bounding_boxes]

                if draw_lines:  # could be changed to if draw_lines and mlflow_tracking:
                    if not mlflow_tracking:
                        logger.warning("MLFlow tracking is not enabled. MLFLow is required to store the images.")
                    else:
                        img = plot_lines(page, geometric_lines, scale_factor=line_detection_params["pdf_scale_factor"])
                        mlflow.log_image(img, f"pages/{filename}_page_{page.number + 1}_lines.png")

            # Create a document-level groundwater entry
            groundwater_entries = GroundwaterInDocument(
                filename=filename,
                borehole_groundwaters=[
                    GroundwatersInBorehole(value) for _, value in sorted(aggregated_groundwater_entries.items())
                ],
            )

            # create list of BoreholePrediction objects with all the separate lists
            borehole_predictions_list: list[BoreholePredictions] = BoreholeListBuilder(
                layers_in_document=layers_in_document,
                file_name=filename,
                groundwater_in_doc=groundwater_entries,
                bounding_boxes=bounding_boxes,
                elevations_list=metadata.elevations,
                coordinates_list=metadata.coordinates,
            ).build()

            # Add file predictions
            predictions.add_file_predictions(FilePredictions(borehole_predictions_list, file_metadata, filename))

            #TODO: Add an approach that stores each borehole in a separate file for files with multiple boreholes
            # Add layers to a csv file
            if csv:
                base_path = out_directory / Path(filename).stem
                csv_list = CsvPredictions(borehole_predictions_list).to_csv()

                for borehole_index, csv_content in enumerate(csv_list):
                    csv_path = f"{base_path}_{borehole_index}.csv" if len(csv_list) > 1 else f"{base_path}.csv"
                    logger.info("Writing CSV predictions to %s", csv_path)
                    with open(csv_path, "w", encoding="utf8", newline='') as file:
                        file.write(csv_content)

                    if mlflow_tracking:
                        mlflow.log_artifact(csv_path, "csv")

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
