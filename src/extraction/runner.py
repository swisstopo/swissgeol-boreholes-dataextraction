"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from tqdm import tqdm

from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_tracking
from core.pipeline_runner import MultiBenchmarkRunner, PipelineRunner, PipelineRunResult
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
from extraction.utils.benchmark_utils import log_metric_mlflow
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")

logger = logging.getLogger(__name__)


def write_json_predictions(path: Path, predictions: OverallFilePredictions) -> None:
    """Write prediction to json output.

    Args:
        path (Path): Destination file.
        predictions (OverallFilePredictions): Prediction to dump in JSON file.
    """
    with open(path, "w", encoding="utf8") as file:
        json.dump(predictions.to_json(), file, ensure_ascii=False, indent=2)


def read_json_predictions(path: Path) -> OverallFilePredictions:
    """Read predictions from input file.

    Returns an empty OverallFilePredictions if the file doesn't exist or contains invalid JSON.

    Args:
        path (Path): File to read and parse.

    Returns:
        OverallFilePredictions: Parsed predictions.
    """
    if not path.exists():
        return OverallFilePredictions()
    try:
        with open(path, encoding="utf8") as f:
            return OverallFilePredictions.from_json(json.load(f))
    except json.JSONDecodeError:
        logger.warning(f"Unable to load prediction from file {path}")
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


@dataclass
class ExtractionOptions:
    """Options shared between single and multi-benchmark extraction runners."""

    skip_draw_predictions: bool = False
    draw_lines: bool = False
    draw_tables: bool = False
    draw_strip_logs: bool = False
    csv: bool = False
    matching_analytics: bool = False
    part: str = "all"


def run_extraction_predictions(
    input_directory: Path,
    out_directory: Path,
    predictions_path_tmp: Path,
    options: ExtractionOptions,
    ground_truth: GroundTruth | None = None,
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
        options (ExtractionOptions): Extraction run options.
        ground_truth (GroundTruth | None): Ground truth for evaluation.
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

    any_draw = (
        not options.skip_draw_predictions or options.draw_lines or options.draw_tables or options.draw_strip_logs
    )

    all_csv_paths: list[Path] = []
    for pdf_file in tqdm(pdf_files, desc="Processing files", unit="file"):
        # Check if file is already computed in previous run
        if predictions.contains(pdf_file.name):
            logger.info(f"{pdf_file.name} already predicted.")
            continue

        logger.info(f"Processing file: {pdf_file.name}")
        try:
            result = extract(file=pdf_file, filename=pdf_file.name, part=options.part, analytics=analytics)
            prediction = evaluate_single_prediction(result.predictions, ground_truth)
            predictions.add_file_predictions(prediction)

            if options.csv:
                all_csv_paths.extend(write_csv_for_file(result.predictions, out_directory))

            if any_draw:
                # Run evaluation for current file drawing
                result.predictions = evaluate_single_prediction(result.predictions, ground_truth)

                draw_file_predictions(
                    result=result,
                    file=pdf_file,
                    filename=pdf_file.name,
                    out_directory=out_directory,
                    skip_draw_predictions=options.skip_draw_predictions,
                    draw_lines=options.draw_lines,
                    draw_tables=options.draw_tables,
                    draw_strip_logs=options.draw_strip_logs,
                )

            logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
            write_json_predictions(path=predictions_path_tmp, predictions=predictions)

        except Exception as e:
            logger.error(f"Unexpected error in file {pdf_file.name}. Trace: {e}")

    return predictions, n_documents, all_csv_paths


_ExtractionResult = tuple[OverallFilePredictions, list[Path]]


@dataclass(kw_only=True)
class ExtractionPipelineRunner(PipelineRunner[_ExtractionResult, ExtractionBenchmarkSummary]):
    """Runs the boreholes data extraction pipeline."""

    input_directory: Path
    ground_truth_path: Path | None
    out_directory: Path
    metadata_path: Path
    options: ExtractionOptions = field(default_factory=ExtractionOptions)
    runname: str | None = None
    analytics: MatchingParamsAnalytics | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.analytics = create_analytics() if self.options.matching_analytics else None
        self.copy_predictions_to_final = self.options.part == "all"
        self.out_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    def setup_mlflow_run(self, runid: str | None) -> str:
        return setup_mlflow_tracking(
            run_id=runid,
            experiment_name="Boreholes data extraction",
            runname=self.runname,
            nested=self.is_nested,
            tags={
                "input_directory": self.input_directory,
                "ground_truth_path": self.ground_truth_path,
                "out_directory": self.out_directory,
                "predictions_path": self.predictions_path,
                "metadata_path": self.metadata_path,
            },
            params={
                **flatten(line_detection_params),
                **flatten(matching_params),
            },
        )

    def run_predictions(self, predictions_path_tmp: Path) -> PipelineRunResult[_ExtractionResult]:
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None
        predictions, n_documents, csv_paths = run_extraction_predictions(
            input_directory=self.input_directory,
            out_directory=self.out_directory,
            predictions_path_tmp=predictions_path_tmp,
            options=self.options,
            ground_truth=ground_truth,
            analytics=self.analytics,
        )
        return PipelineRunResult(
            result=(predictions, csv_paths),
            n_documents=n_documents,
        )

    def evaluate(self, run_result: PipelineRunResult[_ExtractionResult]) -> ExtractionBenchmarkSummary | None:
        predictions, _ = run_result.result
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None
        eval_summary = evaluate_all_predictions(
            predictions=predictions,
            ground_truth=ground_truth,
        )
        if eval_summary is not None:
            eval_summary.n_documents = run_result.n_documents
        return eval_summary

    def after_evaluation(
        self,
        run_result: PipelineRunResult[_ExtractionResult],
        summary: ExtractionBenchmarkSummary | None,
        _predictions_path_tmp: Path,
    ) -> None:
        predictions, csv_paths = run_result.result

        if mlflow:
            for csv_path in csv_paths:
                mlflow.log_artifact(str(csv_path), "csv")

            if summary is not None:
                log_metric_mlflow(summary, out_dir=self.out_directory)

        logger.info(f"Metadata written to {self.metadata_path}")
        with open(self.metadata_path, "w", encoding="utf8") as file:
            json.dump(predictions.get_metadata_as_dict(), file, ensure_ascii=False)

        if self.options.matching_analytics and self.analytics is not None:
            analytics_output_path = self.out_directory / "matching_params_analytics.json"
            self.analytics.save_analytics(analytics_output_path)
            logger.info(f"Matching parameters analytics saved to {analytics_output_path}")

        if self.options.part == "all":
            logger.info(f"Writing predictions to final JSON file {self.predictions_path}")


@dataclass(kw_only=True)
class ExtractionBenchmarkRunner(MultiBenchmarkRunner[BenchmarkSpec, ExtractionBenchmarkSummary]):
    """Orchestrates multiple extraction benchmarks with shared MLflow parent tracking."""

    experiment_name = "Boreholes data extraction"
    input_tag_name = "input_directory"
    input_path_attr = "input_path"
    aggregate_label = "overall"
    runname = "benchmark"

    options: ExtractionOptions = field(default_factory=ExtractionOptions)

    def run_single(self, spec: BenchmarkSpec) -> ExtractionBenchmarkSummary | None:
        logger.info("Running benchmark: %s", spec.name)
        bench_out = self.multi_root / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)

        return ExtractionPipelineRunner(
            predictions_path=bench_out / "predictions.json",
            resume=self.resume,
            is_nested=True,
            input_directory=spec.input_path,
            ground_truth_path=spec.ground_truth_path,
            out_directory=bench_out,
            metadata_path=bench_out / "metadata.json",
            options=self.options,
            runname=spec.name,
        ).execute()
