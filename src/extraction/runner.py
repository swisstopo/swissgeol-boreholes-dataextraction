"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

import json
import logging
import os
import shutil
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
from tqdm import tqdm

from core.mlflow_tracking import mlflow
from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
from extraction.annotations.plot_utils import plot_lines, save_visualization
from extraction.evaluation.benchmark.score import BenchmarkSummary, evaluate_all_predictions
from extraction.evaluation.benchmark.spec import BenchmarkSpec
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
from extraction.utils.benchmark_utils import _parent_input_directory_key, _short_metric_key, log_metric_mlflow
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params
from swissgeol_doc_processing.utils.strip_log_detection import detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import detect_table_structures

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")

logger = logging.getLogger(__name__)

if mlflow:
    import pygit2


def _finalize_overall_summary(
    *,
    overall_results: list[tuple[str, BenchmarkSummary | None]],
    multi_root: Path,
):
    """Write overall benchmark summary.

    Also logs overall aggregate metrics + artifacts to MLflow on the parent run (if enabled).

    Args:
        overall_results (list[tuple[str, BenchmarkSummary]]): List of tuples
            containing (benchmark_name, BenchmarkSummary)
        multi_root (Path): Output path.
    """
    summary_path = multi_root / "overall_summary.csv"
    # Write data as new dataframe entires
    rows = []
    for benchmark, summary in overall_results:
        row: dict[str, Any] = {"benchmark": benchmark}
        if summary is not None:
            row["ground_truth_path"] = summary.ground_truth_path
            row["n_documents"] = summary.n_documents
            row.update(summary.metrics())
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by="benchmark")

    total_docs = df["n_documents"].sum()
    means = df.mean(numeric_only=True)
    agg_row = means.round(3)
    agg_row.at["benchmark"] = "overall"
    agg_row.at["n_documents"] = total_docs
    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    df.to_csv(summary_path, index=False)

    # --- MLflow: overall mean metrics + artifacts on parent run ---
    if mlflow:
        for full_key, value in means.items():
            if pd.notna(value):
                short_key = _short_metric_key(full_key)
                mlflow.log_metric(short_key, value)

        if pd.notna(total_docs):
            mlflow.log_metric("total_n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")


def write_json_predictions(filename: str, predictions: OverallFilePredictions) -> None:
    """Write prediction to json output.

    Args:
        filename (str): Destination file.
        predictions (OverallFilePredictions): Prediction to dump in JSON file.
    """
    with open(filename, "w", encoding="utf8") as file:
        json.dump(predictions.to_json(), file, ensure_ascii=False)


def delete_temporary(pattern: Path) -> None:
    """Delete temporary files matching a glob pattern.

    Only files ending with '.tmp' are deleted.

    Args:
        pattern (Path): Glob pattern to match files (e.g., '/path/*.tmp' or '/path/**/*.tmp').
    """
    for file in glob(str(pattern)):
        if Path(file).suffix == ".tmp":
            os.remove(file)


def read_mlflow_runid(filename: str) -> str | None:
    """Read locally stored mlflow run id.

    Args:
        filename (str): Name of the file that contains runid.

    Returns:
        str | None: Loaded runid if any, otherwise None.
    """
    if not Path(filename).exists():
        return None

    with open(filename, encoding="utf8") as f:
        return json.load(f)


def write_mlflow_runid(filename: str, runid: str) -> None:
    """Locally stores mlflow run id.

    Args:
        filename (str): Name of the file to store runid.
        runid (str): Runid to store.
    """
    with open(filename, "w", encoding="utf8") as file:
        json.dump(runid, file, ensure_ascii=False)


def read_json_predictions(filename: str) -> OverallFilePredictions:
    """Read predictions from input file.

    Returns an empty OverallFilePredictions if the file doesn't exist or contains invalid JSON.

    Args:
        filename (str): File to read and parse.

    Returns:
        OverallFilePredictions: Parsed predictions.
    """
    if not Path(filename).exists():
        return OverallFilePredictions()
    try:
        with open(filename, encoding="utf8") as f:
            return OverallFilePredictions.from_json(json.load(f))
    except json.JSONDecodeError:
        logger.warning(f"Unable to load prediction from file {filename}")
        return OverallFilePredictions()


@contextmanager
def open_pdf(
    file: Path | BytesIO,
    filename: str = None,
) -> Generator[pymupdf.Document, None, None]:
    """Open a PDF document from either a file path or a binary stream.

    This context manager handles opening and closing a PyMuPDF document,
    accepting either a file path or an already-open binary stream.

    Args:
        file (Path | BytesIO): Either a Path to a PDF file or an open binary stream (BytesIO).
        filename (str): Filename to associate with the document. Only used when 'file' is a stream.

    Yields:
        pymupdf.Document: The opened PDF document.
    """
    doc = (
        pymupdf.Document(filename=filename, stream=file)
        if isinstance(file, BytesIO)
        else pymupdf.Document(filename=file)
    )
    yield doc
    doc.close()


def extract(
    file: Path | BytesIO,
    filename: str,
    out_directory: Path | None = None,
    skip_draw_predictions: bool = True,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    part: str = "all",
    analytics: MatchingParamsAnalytics | None = None,
) -> FilePredictions:
    """Extract pipeline for input file `in_path`.

    Args:
        file (Path | BytesIO): Path or stream of file to process.
        filename (str): Name of the file used as identifier.
        out_directory (Path): The directory to store results if any. Defaults to None.
        skip_draw_predictions (bool): Whether to skip drawing predictions on pdf pages. Defaults to True.
        draw_lines (bool): Whether to draw lines on pdf pages. Defaults to False.
        draw_tables (bool): Whether to draw detected table structures on pdf pages. Defaults to False.
        draw_strip_logs (bool): Whether to draw detected strip log structures on pages. Defaults to False.
        csv (bool): Whether to generate a CSV output. Defaults to False.
        part (str): Pipeline mode, "all" for full extraction, "metadata" for metadata only. Defaults to "all".
        analytics (MatchingParamsAnalytics): Analytics object for tracking matching parameters. Defaults to None.

    Returns:
        FilePredictions: Prediction for input file
    """
    draw_directory = None

    if (not skip_draw_predictions or csv) and out_directory is None:
        logger.error("Please provide out directory to save results")
        raise FileNotFoundError()

    if not skip_draw_predictions:
        # check if directories exist and create them when necessary
        draw_directory = out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

    # Clear cache to avoid cache contamination across different files, which can cause incorrect
    # visualizations; see also https://github.com/swisstopo/swissgeol-boreholes-suite/issues/1935
    pymupdf.TOOLS.store_shrink(100)

    with open_pdf(file=file, filename=filename) as doc:
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
            logger.info(f"Processing page {page_number}")

            text_lines = extract_text_lines(page)
            long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
            name_entries = extract_borehole_names(text_lines, name_detection_params)
            all_name_entries.name_feature_list.extend(name_entries)

            # Detect table structures on the page
            table_structures = detect_table_structures(
                page, long_or_horizontal_lines, text_lines, table_detection_params
            )

            # Detect strip logs on the page
            strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

            # Extract the stratigraphy
            page_layers = extract_page(
                text_lines,
                long_or_horizontal_lines,
                all_geometric_lines,
                table_structures,
                strip_logs,
                file_metadata.language,
                page_index,
                page,
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
                save_visualization(img, filename, page.number + 1, "tables", draw_directory)

            # Draw strip logs if requested
            if draw_strip_logs:
                img = plot_strip_logs(page, strip_logs, page_index)
                save_visualization(img, filename, page.number + 1, "strip_logs", draw_directory)

            if draw_lines:
                img = plot_lines(page, all_geometric_lines, scale_factor=line_detection_params["pdf_scale_factor"])
                save_visualization(img, filename, page.number + 1, "lines", draw_directory)

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
            plot_prediction(prediction, doc, draw_directory)

    # Add layers to a csv file
    if csv:
        csv_directory = out_directory / "csv"
        csv_directory.mkdir(parents=True, exist_ok=True)
        base_path = csv_directory / Path(filename).stem

        for index, borehole in enumerate(borehole_predictions_list):
            csv_path = f"{base_path}_{index}.csv" if len(borehole_predictions_list) > 1 else f"{base_path}.csv"
            logger.info(f"Writing CSV predictions to {csv_path}")
            with open(csv_path, "w", encoding="utf8", newline="") as csvfile:
                csvfile.write(borehole.to_csv())

            if mlflow:
                mlflow.log_artifact(csv_path, "csv")

    return prediction


def setup_mlflow_tracking(
    runid: str | None,
    input_directory: Path,
    ground_truth_path: Path,
    runname: str | None = None,
    out_directory: Path = None,
    predictions_path: Path = None,
    metadata_path: Path = None,
    experiment_name: str = "Boreholes data extraction",
    nested: bool = False,
) -> str:
    """Initialize and configure an MLflow run with experiment tags and parameters.

    Args:
        runid (str): Existing run ID to resume, or None to start a new run.
        input_directory (Path): Input directory path to log as MLflow tag.
        ground_truth_path (Path): Ground truth file path to log as MLflow tag.
        runname (str, optional): Run name for MLflow. Defaults to None.
        out_directory (Path, optional): Output directory tracking. Defaults to None.
        predictions_path (Path, optional): Prediction path tracking. Defaults to None.
        metadata_path (Path, optional): Metadata path tracking. Defaults to None.
        experiment_name (str, optional): Experiment name tracking. Defaults to "Boreholes data extraction".
        nested (bool, optional): Indicate if the current run is nested.

    Raises:
        ValueError: If MLFLOW_TRACKING environment variable is not set to "True".

    Returns:
        str: The active MLflow run ID.
    """
    if not mlflow:
        raise ValueError("Tracking is not activated")

    mlflow.set_experiment(experiment_name)

    # only start a run if none is active
    try:
        mlflow.start_run(run_name=runname, run_id=runid, nested=nested)
    except mlflow.MlflowException:
        mlflow.start_run(run_name=runname, nested=nested)
        logger.warning(f"Unable to resume run with ID: {runid} ({runname}), start new one.")

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

    return mlflow.active_run().info.run_id


def _setup_mlflow_parent_run(
    *, benchmarks: Sequence[BenchmarkSpec], runname: str | None = None, runid: str | None = None
) -> str:
    """Start the parent MLflow run (multi-benchmark) and log global params once.

    Args:
        benchmarks (Sequence[BenchmarkSpec]): List of benchmark specs.
        runname (str, optional): Run name to resume if any. Defaults to None.
        runid (str, optional): Run id to resume if any. Defaults to None.

    Raises:
        ValueError: If MLFLOW_TRACKING environment variable is not set to "True".

    Returns:
        str: Current run id to parent run.
    """
    if not mlflow:
        raise ValueError("Tracking is not activated")

    runid = setup_mlflow_tracking(
        runname=runname,
        runid=runid,
        input_directory=_parent_input_directory_key(benchmarks),
        ground_truth_path=None,  # parent has no single GT
    )

    mlflow.set_tag("run_type", "multi_benchmark")
    mlflow.set_tag("benchmarks", ",".join(b.name for b in benchmarks))

    return runid


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
    resume: bool | None = False,
    matching_analytics: bool = False,
    part: str = "all",
    runname: str | None = None,
    is_nested: bool = False,
) -> None | BenchmarkSummary:
    """Run the boreholes data extraction pipeline.

    The pipeline will extract material description of all found layers and assign them to the corresponding
    depth intervals. The input directory should contain pdf files with boreholes data. The algorithm can deal
    with borehole profiles of multiple pages.

    Note: This function is designed to be called from the label-studio backend, whereas the click_pipeline function
    is called from the CLI.

    Args:
        input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
        ground_truth_path (Path | None): The path to the ground truth file json file.
        out_directory (Path): The directory to store the evaluation results.
        predictions_path (Path): The path to the predictions file.
        metadata_path (Path): The path to the metadata file.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.
        draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.
        draw_tables (bool, optional): Whether to draw detected table structures on pdf pages. Defaults to False.
        draw_strip_logs (bool, optional): Whether to draw detected strip log structures on pages. Defaults to False.
        csv (bool): Whether to generate a CSV output. Defaults to False.
        resume (bool, optional): Resume previous run if available. Defaults to false.
        matching_analytics (bool): Whether to enable matching parameters analytics. Defaults to False.
        part (str): Pipeline mode, "all" for full extraction, "metadata" for metadata only. Defaults to "all".
        runname (str, optional): Run name for MLflow. Defaults to None.
        is_nested (bool, optional): If True, indicates this is a nested run (called from benchmark pipeline).

    Returns:
        BenchmarkSummary | None: Evaluation summary if ground truth is provided, otherwise None.
    """  # noqa: D301
    # Check that all given outputs exists
    out_directory.mkdir(exist_ok=True)
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_path_tmp = predictions_path.parent / (predictions_path.name + ".tmp")
    mlflow_runid_tmp = predictions_path.parent / ("mlflow_runid.json.tmp")

    # Clean old run if no resume
    if not resume:
        delete_temporary(predictions_path_tmp)
        delete_temporary(mlflow_runid_tmp)

    metadata_path.parent.mkdir(exist_ok=True)

    # Initialize analytics if enabled
    analytics = create_analytics() if matching_analytics else None
    if mlflow:
        # Load run id if existing, otherwise None
        runid = read_mlflow_runid(filename=mlflow_runid_tmp)
        runid = setup_mlflow_tracking(
            runid=runid,
            input_directory=input_directory,
            ground_truth_path=ground_truth_path,
            runname=runname or input_directory.name,
            out_directory=out_directory,
            predictions_path=predictions_path,
            metadata_path=metadata_path,
            nested=is_nested,
        )
        # Save current run id
        write_mlflow_runid(filename=mlflow_runid_tmp, runid=runid)

    # if a file is specified instead of an input directory, copy the file to a temporary directory and work with that.
    if input_directory.is_file():
        root = input_directory.parent
        files = [input_directory.name]
    else:
        root = input_directory
        _, _, files = next(os.walk(input_directory))

    # Check if tmp file exists with unfinished experiment
    predictions = read_json_predictions(predictions_path_tmp)

    # Iterate over all files
    for filename in tqdm(files, desc="Processing files", unit="file"):
        # Check if file extension is supported
        if not filename.endswith(".pdf"):
            logger.warning(f"{filename} does not end with .pdf and is not treated.")
            continue

        # Check if file already predicted
        if predictions.contains(filename):
            logger.info(f"{filename} already predicted.")
            continue

        in_path = root / filename
        logger.info(f"Processing file: {in_path}")

        try:
            # Add file predictions
            prediction = extract(
                file=in_path,
                filename=in_path.name,
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

            # Track progress in tmp file
            logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
            write_json_predictions(filename=predictions_path_tmp, predictions=predictions)

        except Exception as e:
            logger.error(f"Unexpected error in file {filename}. Trace: {e}")

    # Evaluate final predictions
    eval_summary = evaluate_all_predictions(
        predictions=predictions,
        ground_truth_path=ground_truth_path,
    )

    # Log metrics to MLflow if enabled
    if mlflow and eval_summary is not None:
        log_metric_mlflow(eval_summary, out_dir=out_directory)

    # Save all metadata, analytics and predictions (if needed)
    logger.info(f"Metadata written to {metadata_path}")
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(predictions.get_metadata_as_dict(), file, ensure_ascii=False)

    # Finalize analytics if enabled
    if matching_analytics:
        # Warning: Resuming analytics is not supported
        analytics_output_path = out_directory / "matching_params_analytics.json"
        analytics.save_analytics(analytics_output_path)
        logger.info(f"Matching parameters analytics saved to {analytics_output_path}")

    # Track progress to finale file and remove tmp
    if part == "all":
        logger.info(f"Writing predictions to final JSON file {predictions_path}")
        shutil.copy(src=predictions_path_tmp, dst=predictions_path)

    # Clean temporary files only if not nested
    if not is_nested:
        delete_temporary(predictions_path_tmp)
        delete_temporary(mlflow_runid_tmp)

    # Terminate runid
    if mlflow:
        mlflow.end_run()

    return eval_summary


def start_pipeline_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    resume: bool = False,
    matching_analytics: bool = False,
    part: str = "all",
) -> None:
    """Run multiple benchmarks in one execution.

    Output is namespaced per benchmark under:
      <out_directory>/<benchmark_name>/

    Args:
        benchmarks (Sequence[BenchmarkSpec]): List of benchmark specifications.
        out_directory (Path): Output directory for multi-benchmark results.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions. Defaults to False.
        draw_lines (bool, optional): Whether to draw detected lines. Defaults to False.
        draw_tables (bool, optional): Whether to draw detected tables. Defaults to False.
        draw_strip_logs (bool, optional): Whether to draw strip logs. Defaults to False.
        csv (bool, optional): Whether to output CSV summaries. Defaults to False.
        resume (bool, optional): Resume previous run if available. Defaults to false.
        matching_analytics (bool, optional): Whether to compute matching analytics. Defaults to False.
        part (str): Pipeline mode, "all" for full extraction, "metadata" for metadata only. Defaults to "all".
    """
    # Create root directory for multi-benchmarking
    out_directory.mkdir(parents=True, exist_ok=True)

    # Create temp file for parent run
    mlflow_runid_tmp = out_directory / "mlflow_parent_runid.json.tmp"

    if mlflow:
        # Load run id if existing, otherwise None
        parent_runid = read_mlflow_runid(filename=mlflow_runid_tmp) if resume else None
        parent_runid = _setup_mlflow_parent_run(benchmarks=benchmarks, runname="benchmark", runid=parent_runid)
        # Save current run id
        write_mlflow_runid(filename=mlflow_runid_tmp, runid=parent_runid)

    overall_results = []
    for spec in benchmarks:
        # Create bench folder to outputs
        bench_out = out_directory / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)
        bench_predictions_path = bench_out / "predictions.json"
        bench_metadata_path = bench_out / "metadata.json"

        eval_result = start_pipeline(
            input_directory=spec.input_path,
            ground_truth_path=spec.ground_truth_path,
            out_directory=bench_out,
            predictions_path=bench_predictions_path,
            metadata_path=bench_metadata_path,
            skip_draw_predictions=skip_draw_predictions,
            draw_lines=draw_lines,
            draw_tables=draw_tables,
            draw_strip_logs=draw_strip_logs,
            csv=csv,
            resume=resume,
            matching_analytics=matching_analytics,
            part=part,
            runname=spec.name,
            is_nested=True,
        )
        overall_results.append((spec.name, eval_result))

    # Write aggregation
    _finalize_overall_summary(
        overall_results=overall_results,
        multi_root=out_directory,
    )

    # Clean temporary files
    delete_temporary(mlflow_runid_tmp)  # Main run
    delete_temporary(out_directory / "*" / "*.tmp")  # Nested runs
