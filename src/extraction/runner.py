"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

import json
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from tqdm import tqdm

from core.benchmark_utils import (
    _short_metric_key,
    finalize_overall_summary,
    finalize_pipeline_run,
    parent_input_key,
    prepare_pipeline_temp_paths,
    run_multi_benchmark,
    start_or_resume_mlflow_run,
)
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_parent_run, setup_mlflow_tracking
from extraction.core.extract import ExtractionResult, extract, open_pdf
from extraction.evaluation.benchmark.score import ExtractionBenchmarkSummary, evaluate_all_predictions
from extraction.evaluation.benchmark.spec import BenchmarkSpec
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.utils.benchmark_utils import log_metric_mlflow
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")

logger = logging.getLogger(__name__)


def write_json_predictions(filename: str, predictions: OverallFilePredictions) -> None:
    """Write prediction to json output.

    Args:
        filename (str): Destination file.
        predictions (OverallFilePredictions): Prediction to dump in JSON file.
    """
    with open(filename, "w", encoding="utf8") as file:
        json.dump(predictions.to_json(), file, ensure_ascii=False)


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


def setup_extraction_mlflow_tracking(
    runid: str | None,
    input_directory: Path,
    ground_truth_path: Path | None,
    runname: str | None = None,
    out_directory: Path | None = None,
    predictions_path: Path | None = None,
    metadata_path: Path | None = None,
    experiment_name: str = "Boreholes data extraction",
    nested: bool = False,
) -> str:
    """Wraps setup_mlflow_tracking() with extraction-specific tags and params."""
    return setup_mlflow_tracking(
        run_id=runid,
        experiment_name=experiment_name,
        runname=runname,
        nested=nested,
        tags={
            "input_directory": input_directory,
            "ground_truth_path": ground_truth_path,
            "out_directory": out_directory,
            "predictions_path": predictions_path,
            "metadata_path": metadata_path,
        },
        params={
            **flatten(line_detection_params),
            **flatten(matching_params),
        },
        include_git_info=True,
    )


def _setup_mlflow_parent_run(
    *, benchmarks: Sequence[BenchmarkSpec], runname: str | None = None, runid: str | None = None
) -> str:
    """Wraps setup_mlflow_parent_run() with extraction specific input key and tags."""
    return setup_mlflow_parent_run(
        run_id=runid,
        experiment_name="Boreholes data extraction",
        runname=runname,
        parent_input_key=parent_input_key([Path(b.input_path) for b in benchmarks]),
        benchmarks=benchmarks,
        input_tag_name="input_directory",
        ground_truth_path=None,
        include_git_info=True,
    )


def run_predictions(
    input_directory: Path,
    out_directory: Path,
    predictions_path_tmp: Path,
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
    if input_directory.is_file():
        root = input_directory.parent
        pdf_files = [input_directory.name] if input_directory.suffix.lower() == ".pdf" else []
    else:
        root = input_directory
        pdf_files = [f.name for f in input_directory.glob("*.pdf") if f.is_file()]

    n_documents = len(pdf_files)

    # Load any partially-completed predictions for resume support
    predictions = read_json_predictions(predictions_path_tmp)

    any_draw = not skip_draw_predictions or draw_lines or draw_tables or draw_strip_logs

    all_csv_paths: list[Path] = []

    for filename in tqdm(pdf_files, desc="Processing files", unit="file"):
        if predictions.contains(filename):
            logger.info(f"{filename} already predicted.")
            continue

        in_path = root / filename
        logger.info(f"Processing file: {in_path}")

        try:
            result = extract(file=in_path, filename=in_path.name, part=part, analytics=analytics)
            predictions.add_file_predictions(result.predictions)

            if csv:
                all_csv_paths.extend(write_csv_for_file(result.predictions, out_directory))

            if any_draw:
                draw_file_predictions(
                    result=result,
                    file=in_path,
                    filename=in_path.name,
                    out_directory=out_directory,
                    skip_draw_predictions=skip_draw_predictions,
                    draw_lines=draw_lines,
                    draw_tables=draw_tables,
                    draw_strip_logs=draw_strip_logs,
                )

            logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
            write_json_predictions(filename=predictions_path_tmp, predictions=predictions)

        except Exception as e:
            logger.error(f"Unexpected error in file {filename}. Trace: {e}")

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
    temp_paths = prepare_pipeline_temp_paths(
        predictions_path,
        resume=bool(resume),
        cleanup_mlflow_tmp=True,
    )
    predictions_path_tmp = temp_paths.predictions_path_tmp
    mlflow_runid_tmp = temp_paths.mlflow_runid_tmp

    metadata_path.parent.mkdir(exist_ok=True)

    # Initialize analytics if enabled
    analytics = create_analytics() if matching_analytics else None

    start_or_resume_mlflow_run(
        resume=bool(resume),
        mlflow_runid_tmp=mlflow_runid_tmp,
        setup_run=lambda runid: setup_extraction_mlflow_tracking(
            runid=runid,
            input_directory=input_directory,
            ground_truth_path=ground_truth_path,
            runname=runname or input_directory.name,
            out_directory=out_directory,
            predictions_path=predictions_path,
            metadata_path=metadata_path,
            nested=is_nested,
        ),
    )

    predictions, n_documents, csv_paths = run_predictions(
        input_directory=input_directory,
        out_directory=out_directory,
        predictions_path_tmp=predictions_path_tmp,
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

    # Evaluate final predictions
    eval_summary = evaluate_all_predictions(
        predictions=predictions,
        ground_truth_path=ground_truth_path,
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

    if part == "all":
        logger.info(f"Writing predictions to final JSON file {predictions_path}")

    finalize_pipeline_run(
        is_nested=is_nested,
        predictions_path_tmp=predictions_path_tmp,
        final_predictions_path=predictions_path,
        copy_predictions=(part == "all"),
        mlflow_runid_tmp=mlflow_runid_tmp,
    )

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
    """Run multiple benchmarks in one execution."""
    parent_runid_tmp = out_directory / "mlflow_parent_runid.json.tmp"

    def setup_parent_run(parent_runid: str | None) -> str:
        return _setup_mlflow_parent_run(
            benchmarks=benchmarks,
            runname="benchmark",
            runid=parent_runid,
        )

    def run_single(spec: BenchmarkSpec) -> ExtractionBenchmarkSummary | None:
        logger.info("Running benchmark: %s", spec.name)

        bench_out = out_directory / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)
        bench_predictions_path = bench_out / "predictions.json"
        bench_metadata_path = bench_out / "metadata.json"

        return start_pipeline(
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

    def finalize(
        overall_results: list[tuple[str, ExtractionBenchmarkSummary | None]],
        root: Path,
    ) -> None:
        finalize_overall_summary(
            overall_results=overall_results,
            multi_root=root,
            aggregate_label="overall",
            metric_key_shortener=_short_metric_key,
        )

    run_multi_benchmark(
        benchmarks=benchmarks,
        multi_root=out_directory,
        resume=resume,
        parent_runid_tmp=parent_runid_tmp,
        setup_parent_run=setup_parent_run if mlflow else None,
        run_single_benchmark=run_single,
        finalize_summary=finalize,
    )
