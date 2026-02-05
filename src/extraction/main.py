"""This module contains the main pipeline for the boreholes data extraction."""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from extraction.evaluation.benchmark.spec import parse_benchmark_spec
from extraction.runner import start_pipeline, start_pipeline_benchmark
from extraction.utils.benchmark_utils import configure_logging
from swissgeol_doc_processing.utils.file_utils import get_data_path, read_params

load_dotenv()
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
        required=False,  # make it optional to allow multi-benchmarks
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
        default=get_data_path() / "output",
        help="Path to the output directory.",
    )(f)
    f = click.option(
        "-p",
        "--predictions-path",
        type=click.Path(path_type=Path),
        default=get_data_path() / "output" / "predictions.json",
        help="Path to the predictions file.",
    )(f)
    f = click.option(
        "-m",
        "--metadata-path",
        type=click.Path(path_type=Path),
        default=get_data_path() / "output" / "metadata.json",
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


@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    help="Repeatable benchmark spec: '<name>:<input_path>:<ground_truth_path>'. "
    "If provided, runs multiple benchmarks in one execution.",
)
@click.command()
@common_options
@click.option(
    "-pa", "--part", type=click.Choice(["all", "metadata"]), default="all", help="The part of the pipeline to run."
)
def click_pipeline(
    input_directory: Path | None,
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
    benchmarks: tuple[str, ...] = (),
):
    """Run the boreholes data extraction pipeline."""
    # Setup logging (same for all)
    configure_logging()

    # --- Multi-benchmark mode ---
    if benchmarks:
        specs = [parse_benchmark_spec(b) for b in benchmarks]

        start_pipeline_benchmark(
            benchmarks=specs,
            out_directory=out_directory,
            skip_draw_predictions=skip_draw_predictions,
            draw_lines=draw_lines,
            draw_tables=draw_tables,
            draw_strip_logs=draw_strip_logs,
            csv=csv,
            matching_analytics=matching_analytics,
            part=part,
        )
    # --- Single-benchmark mode ---
    else:
        # If no multi-benchmarking, enforce -i argument
        if input_directory is None:
            raise click.BadParameter("Missing -i/--input-directory. Provide it, or use one or more --benchmark specs.")
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
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
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
        draw_tables=draw_tables,
        draw_strip_logs=draw_strip_logs,
        csv=csv,
        matching_analytics=matching_analytics,
        part="metadata",
    )


if __name__ == "__main__":
    click_pipeline()
