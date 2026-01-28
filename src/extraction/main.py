"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import os
from pathlib import Path

import click
import pymupdf
from dotenv import load_dotenv
from tqdm import tqdm

from extraction import DATAPATH
from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
from extraction.annotations.plot_utils import plot_lines, save_visualization
from extraction.evaluation.benchmark.score import evaluate_all_predictions
from extraction.features.extract import extract_page
from extraction.features.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwaterLevelExtractor,
)
from extraction.features.metadata.borehole_name_extraction import NameInDocument, extract_borehole_names
from extraction.features.metadata.metadata import FileMetadata, MetadataInDocument
from extraction.features.predictions.borehole_predictions import BoreholePredictions
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.predictions.predictions import BoreholeListBuilder
from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import LayersInDocument
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params
from swissgeol_doc_processing.utils.strip_log_detection import detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import detect_table_structures

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow
    import pygit2

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")


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
        "-t",
        "--draw-tables",
        is_flag=True,
        default=False,
        help="Whether to draw detected table structures on pdf pages. Defaults to False.",
    )(f)
    f = click.option(
        "-sl",
        "--draw-strip-logs",
        is_flag=True,
        default=False,
        help="Whether to draw detected strip log structures on pdf pages. Defaults to False.",
    )(f)
    f = click.option(
        "-c",
        "--csv",
        is_flag=True,
        default=False,
        help="Whether to generate CSV output. Defaults to False.",
    )(f)
    f = click.option(
        "-ma",
        "--matching-analytics",
        is_flag=True,
        default=False,
        help="Whether to enable matching parameters analytics. Defaults to False.",
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
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    matching_analytics: bool = False,
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
        draw_tables=draw_tables,
        draw_strip_logs=draw_strip_logs,
        csv=csv,
        matching_analytics=matching_analytics,
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
    matching_analytics: bool = False,
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
        matching_analytics=matching_analytics,
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


def extract(
    in_path: Path,
    out_directory: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    part: str = "all",
    analytics: MatchingParamsAnalytics | None = None,
) -> FilePredictions:
    """Extract pipeline for input file `in_path`.

    Args:
        in_path (Path): Path to file to process.
        out_directory (Path): The directory to store the evaluation results.
        skip_draw_predictions (bool): Whether to skip drawing predictions on pdf pages. Defaults to False.
        draw_lines (bool): Whether to draw lines on pdf pages. Defaults to False.
        draw_tables (bool): Whether to draw detected table structures on pdf pages. Defaults to False.
        draw_strip_logs (bool): Whether to draw detected strip log structures on pages. Defaults to False.
        csv (bool): Whether to generate a CSV output. Defaults to False.
        part (str): The part of the pipeline to run. Defaults to "all".
        analytics (MatchingParamsAnalytics): Analytics object for tracking matching parameters. Defaults to None.

    Returns:
        FilePredictions: Prediction for input file
    """
    filename = in_path.name
    draw_directory = None

    if not skip_draw_predictions:
        # check if directories exist and create them when necessary
        draw_directory = out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

    with pymupdf.Document(in_path) as doc:
        # Extract metadata
        file_metadata = FileMetadata.from_document(doc, matching_params)
        metadata = MetadataInDocument.from_document(doc, file_metadata.language, matching_params)

        # Save the predictions to the overall predictions object, initialize common variables
        all_groundwater_entries = GroundwaterInDocument([], filename)
        all_name_entries = NameInDocument([], filename)
        boreholes_per_page = []

        if part != "all":
            return FilePredictions([], file_metadata, filename)

        # Extract the layers
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            logger.info("Processing page %s", page_number)

            text_lines = extract_text_lines(page)
            long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
            name_entries = extract_borehole_names(text_lines, name_detection_params)
            all_name_entries.name_feature_list.extend(name_entries)

            # Detect table structures on the page
            table_structures = detect_table_structures(
                page_index, doc, long_or_horizontal_lines, text_lines, table_detection_params
            )

            # Detect strip logs on the page
            strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

            # extract the statigraphy
            page_layers = extract_page(
                text_lines,
                long_or_horizontal_lines,
                all_geometric_lines,
                table_structures,
                strip_logs,
                file_metadata.language,
                page_index,
                doc,
                line_detection_params,
                analytics,
                **matching_params,
            )
            boreholes_per_page.append(page_layers)

            # Extract the groundwater levels
            groundwater_extractor = GroundwaterLevelExtractor(file_metadata.language, matching_params)
            groundwater_entries = groundwater_extractor.extract_groundwater(
                page_number=page_number,
                text_lines=text_lines,
                geometric_lines=long_or_horizontal_lines,
                extracted_boreholes=page_layers,
            )
            all_groundwater_entries.groundwater_feature_list.extend(groundwater_entries)

            # Check if need to skip drawing
            if skip_draw_predictions:
                continue

            # Draw table structures if requested
            if draw_tables:
                img = plot_tables(page, table_structures, page_index)
                save_visualization(img, filename, page.number + 1, "tables", draw_directory, mlflow_tracking)

            # Draw strip logs if requested
            if draw_strip_logs:
                img = plot_strip_logs(page, strip_logs, page_index)
                save_visualization(img, filename, page.number + 1, "strip_logs", draw_directory, mlflow_tracking)

            if draw_lines:  # could be changed to if draw_lines and mlflow_tracking:
                img = plot_lines(page, all_geometric_lines, scale_factor=line_detection_params["pdf_scale_factor"])
                save_visualization(img, filename, page.number + 1, "lines", draw_directory, mlflow_tracking)

    # Merge detections if possible
    layers_with_bb_in_document = LayersInDocument(merge_boreholes(boreholes_per_page, matching_params), filename)

    # create list of BoreholePrediction objects with all the separate lists
    borehole_predictions_list: list[BoreholePredictions] = BoreholeListBuilder(
        layers_with_bb_in_document=layers_with_bb_in_document,
        file_name=filename,
        groundwater_in_doc=all_groundwater_entries,
        names_in_doc=all_name_entries,
        elevations_list=metadata.elevations,
        coordinates_list=metadata.coordinates,
    ).build()

    # now that the matching is done, duplicated groundwater can be removed and depths info can be set
    for borehole in borehole_predictions_list:
        borehole.filter_groundwater_entries()

    # Get prediction file
    prediction = FilePredictions(borehole_predictions_list, file_metadata, filename)

    if not skip_draw_predictions:
        # Draw current file prediction
        plot_prediction(prediction, in_path, draw_directory)

    # Add layers to a csv file
    if csv:
        csv_directory = out_directory / "csv"
        csv_directory.mkdir(parents=True, exist_ok=True)
        base_path = csv_directory / Path(filename).stem

        for index, borehole in enumerate(borehole_predictions_list):
            csv_path = f"{base_path}_{index}.csv" if len(borehole_predictions_list) > 1 else f"{base_path}.csv"
            logger.info("Writing CSV predictions to %s", csv_path)
            with open(csv_path, "w", encoding="utf8", newline="") as file:
                file.write(borehole.to_csv())

            if mlflow_tracking:
                mlflow.log_artifact(csv_path, "csv")

    return prediction


def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    matching_analytics: bool = False,
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
        draw_tables (bool, optional): Whether to draw detected table structures on pdf pages. Defaults to False.
        draw_strip_logs (bool, optional): Whether to draw detected strip log structures on pages. Defaults to False.
        metadata_path (Path): The path to the metadata file.
        csv (bool): Whether to generate a CSV output. Defaults to False.
        matching_analytics (bool): Whether to enable matching parameters analytics. Defaults to False.
        part (str, optional): The part of the pipeline to run. Defaults to "all".
    """  # noqa: D301
    # Check that all given outputs exists
    out_directory.mkdir(exist_ok=True)
    predictions_path.parent.mkdir(exist_ok=True)
    metadata_path.parent.mkdir(exist_ok=True)

    # Initialize analytics if enabled
    analytics = create_analytics() if matching_analytics else None

    if mlflow_tracking:
        setup_mlflow_tracking(input_directory, ground_truth_path, out_directory, predictions_path, metadata_path)

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
            logger.warning(f"{filename} does not end with .pdf and is not treated.")
            continue

        in_path = root / filename
        logger.info(f"Processing file: {in_path}")

        try:
            # Add file predictions
            prediction = extract(
                in_path=in_path,
                out_directory=out_directory,
                skip_draw_predictions=skip_draw_predictions,
                draw_lines=draw_lines,
                draw_tables=draw_tables,
                draw_strip_logs=draw_strip_logs,
                csv=csv,
                part=part,
                analytics=analytics,
            )
            predictions.add_file_predictions(prediction)
        except Exception as e:
            logger.error(f"Unexpected error in file {filename}. Trace: {e}")

    logger.info("Metadata written to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(predictions.get_metadata_as_dict(), file, ensure_ascii=False)

    if part == "all":
        logger.info("Writing predictions to JSON file %s", predictions_path)
        with open(predictions_path, "w", encoding="utf8") as file:
            json.dump(predictions.to_json(), file, ensure_ascii=False)

    evaluate_all_predictions(predictions=predictions, ground_truth_path=ground_truth_path)

    # Finalize analytics if enabled
    if matching_analytics:
        analytics_output_path = out_directory / "matching_params_analytics.json"
        analytics.save_analytics(analytics_output_path)
        logger.info(f"Matching parameters analytics saved to {analytics_output_path}")


if __name__ == "__main__":
    click_pipeline()
