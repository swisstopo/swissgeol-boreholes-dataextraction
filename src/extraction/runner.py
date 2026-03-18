"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

import json
import logging
import shutil
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from core.benchmark_utils import (
    _short_metric_key,
    delete_temporary,
    read_mlflow_runid,
    write_mlflow_runid,
)
from core.mlflow_tracking import mlflow
from extraction.core.extract import ExtractionResult, extract, open_pdf
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.benchmark.score import (
    ExtractionBenchmarkSummary,
    evaluate_all_predictions,
    evaluate_single_prediction,
)
from extraction.evaluation.benchmark.spec import BenchmarkSpec
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.utils.benchmark_utils import _parent_input_directory_key_extraction, log_metric_mlflow
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params

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
    overall_results: list[tuple[str, ExtractionBenchmarkSummary | None]],
    multi_root: Path,
):
    """Write overall benchmark summary.

    Also logs overall aggregate metrics + artifacts to MLflow on the parent run (if enabled).

    Args:
        overall_results (list[tuple[str, ExtractionBenchmarkSummary]]): List of tuples
            containing (benchmark_name, ExtractionBenchmarkSummary)
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
    means = df.drop(columns=["n_documents"], errors="ignore").mean(numeric_only=True)
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
            mlflow.log_metric("n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")


def write_json_predictions(filename: str, predictions: OverallFilePredictions) -> None:
    """Write prediction to json output.

    Args:
        filename (str): Destination file.
        predictions (OverallFilePredictions): Prediction to dump in JSON file.
    """
    with open(filename, "w", encoding="utf8") as file:
        json.dump(predictions.to_json(), file, ensure_ascii=False, indent=2)


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


def draw_file_predictions(
    result: ExtractionResult,
    file: Path | BytesIO,
    filename: str,
    out_directory: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
) -> None:
    """Draw extraction visualizations for a single file. Separate from extraction logic.

    Re-opens the PDF to render annotated prediction overlays. Per-page intermediate detection
    data (lines, tables, strip logs) is read from `result.pages_data`. MLflow image logging is
    handled inside save_visualization() when MLFLOW_TRACKING is set.

    Args:
        result (ExtractionResult): Output of extract() for this file.
        file (Path | BytesIO): Original PDF path or stream (re-opened for annotation rendering).
        filename (str): Name of the file used as identifier.
        out_directory (Path): Directory under which a "draw/" sub-folder is created.
        skip_draw_predictions (bool): Skip drawing final prediction overlays. Defaults to False.
        draw_lines (bool): Draw detected lines. Defaults to False.
        draw_tables (bool): Draw detected table structures. Defaults to False.
        draw_strip_logs (bool): Draw detected strip log structures. Defaults to False.
    """
    from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
    from extraction.annotations.plot_utils import plot_lines, save_visualization

    draw_directory = out_directory / "draw"
    draw_directory.mkdir(parents=True, exist_ok=True)

    with open_pdf(file=file, filename=filename) as doc:
        for page_data in result.pages_data:
            page = doc[page_data.page_index]
            page_number = page_data.page_index + 1

            if draw_tables:
                img = plot_tables(page, page_data.table_structures, page_data.page_index)
                save_visualization(img, filename, page_number, "tables", draw_directory)

            if draw_strip_logs:
                img = plot_strip_logs(page, page_data.strip_logs, page_data.page_index)
                save_visualization(img, filename, page_number, "strip_logs", draw_directory)

            if draw_lines:
                img = plot_lines(page, page_data.lines, scale_factor=line_detection_params["pdf_scale_factor"])
                save_visualization(img, filename, page_number, "lines", draw_directory)

        if not skip_draw_predictions:
            plot_prediction(result.predictions, doc, draw_directory)


def write_csv_for_file(predictions: FilePredictions, out_directory: Path) -> list[Path]:
    """Write per-borehole CSV files for a single file's predictions.

    Args:
        predictions (FilePredictions): Predictions for a single file.
        out_directory (Path): Directory under which a "csv/" sub-folder is created.

    Returns:
        list[Path]: Paths of the written CSV files.
    """
    csv_directory = out_directory / "csv"
    csv_directory.mkdir(parents=True, exist_ok=True)
    base_path = csv_directory / Path(predictions.file_name).stem
    csv_paths = []
    for index, borehole in enumerate(predictions.borehole_predictions_list):
        csv_path = (
            Path(f"{base_path}_{index}.csv")
            if len(predictions.borehole_predictions_list) > 1
            else Path(f"{base_path}.csv")
        )
        logger.info(f"Writing CSV predictions to {csv_path}")
        with open(csv_path, "w", encoding="utf8", newline="") as csvfile:
            csvfile.write(borehole.to_csv())
        csv_paths.append(csv_path)
    return csv_paths


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
        input_directory=_parent_input_directory_key_extraction(benchmarks),
        ground_truth_path=None,  # parent has no single GT
    )

    mlflow.set_tag("run_type", "multi_benchmark")
    mlflow.set_tag("benchmarks", ",".join(b.name for b in benchmarks))

    return runid


def run_predictions(
    input_directory: Path,
    out_directory: Path,
    predictions_path_tmp: Path,
    ground_truth: GroundTruth | None = None,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    part: str = "all",
    analytics: MatchingParamsAnalytics | None = None,
) -> tuple[OverallFilePredictions, int, list[Path]]:
    """Discover PDF files, run extract() on each, and write incremental predictions.

    This is the core prediction logic, decoupled from tracking and evaluation.

    Resume is supported: if `predictions_path_tmp` already contains partial results from a previous
    run, those files are skipped and only new files are processed.

    Args:
        input_directory (Path): Directory of PDF files to process, or path to a single PDF file.
        out_directory (Path): Directory where per-file output (visualizations, CSV) is written.
        predictions_path_tmp (Path): Path to the incremental tmp predictions file. Existing content
            is used to resume; the file is updated after each successfully processed file.
        ground_truth (GroundTruth | None): Ground truth for evaluation.
        skip_draw_predictions (bool, optional): Skip drawing predictions on PDF pages. Defaults to False.
        draw_lines (bool, optional): Draw detected lines on PDF pages. Defaults to False.
        draw_tables (bool, optional): Draw detected table structures on PDF pages. Defaults to False.
        draw_strip_logs (bool, optional): Draw detected strip log structures on pages. Defaults to False.
        csv (bool, optional): Generate CSV output for each file. Defaults to False.
        part (str, optional): Pipeline mode: "all" for full extraction, "metadata" for metadata only.
            Defaults to "all".
        analytics (MatchingParamsAnalytics | None, optional): Analytics object for tracking matching
            parameters. Defaults to None.

    Returns:
        tuple[OverallFilePredictions, int, list[Path]]: All predictions accumulated across files,
            the total number of PDF files discovered (including any already-predicted files from a
            resumed run), and all CSV file paths written during this run.
    """
    # Look for files to process
    pdf_files = [input_directory] if input_directory.is_file() else list(input_directory.glob("*.pdf"))
    n_documents = len(pdf_files)

    # Load any partially-completed predictions for resume support
    predictions = read_json_predictions(predictions_path_tmp)

    any_draw = not skip_draw_predictions or draw_lines or draw_tables or draw_strip_logs

    all_csv_paths: list[Path] = []
    for pdf_file in tqdm(pdf_files, desc="Processing files", unit="file"):
        # Check if file is already computed in previous run
        if predictions.contains(pdf_file.name):
            logger.info(f"{pdf_file.name} already predicted.")
            continue

        logger.info(f"Processing file: {pdf_file.name}")

        # Run extraction and happend to predictions
        result = extract(file=pdf_file, filename=pdf_file.name, part=part, analytics=analytics)
        predictions.add_file_predictions(result.predictions)

        if csv:
            all_csv_paths.extend(write_csv_for_file(result.predictions, out_directory))

        if any_draw:
            # Run evaluation for current file drawing
            result.predictions = evaluate_single_prediction(result.predictions, ground_truth)
            # Draw predictions for file
            draw_file_predictions(
                result=result,
                file=pdf_file,
                filename=pdf_file.name,
                out_directory=out_directory,
                skip_draw_predictions=skip_draw_predictions,
                draw_lines=draw_lines,
                draw_tables=draw_tables,
                draw_strip_logs=draw_strip_logs,
            )

        logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
        write_json_predictions(filename=predictions_path_tmp, predictions=predictions)

    return predictions, n_documents, all_csv_paths


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
) -> None | ExtractionBenchmarkSummary:
    """Run the boreholes data extraction pipeline.

    The pipeline will extract material description of all found layers and assign them to the corresponding
    depth intervals. The input directory should contain pdf files with boreholes data. The algorithm can deal
    with borehole profiles of multiple pages.

    Wraps `run_predictions()` with MLflow tracking, evaluation, and analytics.

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
        ExtractionBenchmarkSummary | None: Evaluation summary if ground truth is provided, otherwise None.
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

    # Build ground truth
    ground_truth: GroundTruth | None = None
    if ground_truth_path and ground_truth_path.exists():  # for inference no ground truth is available
        ground_truth = GroundTruth(ground_truth_path)

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

    predictions, n_documents, csv_paths = run_predictions(
        input_directory=input_directory,
        out_directory=out_directory,
        predictions_path_tmp=predictions_path_tmp,
        ground_truth=ground_truth,
        skip_draw_predictions=skip_draw_predictions,
        draw_lines=draw_lines,
        draw_tables=draw_tables,
        draw_strip_logs=draw_strip_logs,
        csv=csv,
        part=part,
        analytics=analytics,
    )

    if mlflow:
        mlflow.log_metric("n_documents", float(n_documents))
        for csv_path in csv_paths:
            mlflow.log_artifact(str(csv_path), "csv")

    # Evaluate final predictions with all data
    eval_summary = evaluate_all_predictions(
        predictions=predictions,
        ground_truth=ground_truth,
    )
    if eval_summary is not None:
        eval_summary.n_documents = n_documents

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
