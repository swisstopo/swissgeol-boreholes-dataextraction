"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_tracking
from core.pipeline_runner import MultiBenchmarkRunner, PipelineRunner, PipelineRunResult
from extraction.core.extract import ExtractionResult, extract
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.benchmark.score import (
    ExtractionBenchmarkSummary,
    evaluate_all_predictions,
    evaluate_prediction,
)
from extraction.evaluation.benchmark.spec import BenchmarkSpec
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


@dataclass
class ExtractionOptions:
    """Options for the extraction runner."""

    matching_analytics: bool = False
    part: str = "all"


@dataclass(kw_only=True)
class ExtractionPipelineRunner(PipelineRunner[OverallFilePredictions, ExtractionBenchmarkSummary]):
    """Runs the boreholes data extraction pipeline."""

    input_directory: Path
    ground_truth_path: Path | None
    out_directory: Path
    metadata_path: Path
    options: ExtractionOptions = field(default_factory=ExtractionOptions)
    on_file_done: Callable[[ExtractionResult, Path, Path], None] | None = None
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

    def run_predictions(self, predictions_path_tmp: Path) -> PipelineRunResult[OverallFilePredictions]:
        """Discover PDF files, run extract() on each, and write incremental predictions.

        This is the core prediction logic, decoupled from tracking and evaluation.

        Resume is supported: if `predictions_path_tmp` already contains partial results from a previous
        run, those files are skipped and only new files are processed.

        Args:
            predictions_path_tmp (Path):  Path to the incremental tmp predictions file. Existing content
                is used to resume; the file is updated after each successfully processed file.

        Returns:
            PipelineRunResult[OverallFilePredictions]: All predictions accumulated across files and the total
                number of PDF files discovered (including any already-predicted files from a resumed run).
        """
        # Look for files to process
        pdf_files: list[Path] = (
            [self.input_directory] if self.input_directory.is_file() else list(self.input_directory.glob("*.pdf"))
        )
        n_documents = len(pdf_files)
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None

        # Load any partially-completed predictions for resume support
        predictions = read_json_predictions(predictions_path_tmp)

        for pdf_file in tqdm(pdf_files, desc="Processing files", unit="file"):
            # Check if file is already computed in previous run
            if predictions.contains(pdf_file.name):
                logger.info(f"{pdf_file.name} already predicted.")
                continue

            logger.info(f"Processing file: {pdf_file.name}")

            result = extract(file=pdf_file, filename=pdf_file.name, part=self.options.part, analytics=self.analytics)
            prediction_with_metrics = evaluate_prediction(result.predictions, ground_truth)
            predictions.add_file_predictions(prediction_with_metrics)

            if self.on_file_done is not None:
                # Pass updated results to callback
                self.on_file_done(result, self.out_directory, pdf_file)

            logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
            write_json_predictions(path=predictions_path_tmp, predictions=predictions)

        return PipelineRunResult(result=predictions, n_documents=n_documents)

    def evaluate(self, run_result: PipelineRunResult[OverallFilePredictions]) -> ExtractionBenchmarkSummary | None:
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None
        eval_summary = evaluate_all_predictions(
            predictions=run_result.result,
            ground_truth=ground_truth,
        )
        if eval_summary is not None:
            eval_summary.n_documents = run_result.n_documents
        return eval_summary

    def after_evaluation(
        self,
        run_result: PipelineRunResult[OverallFilePredictions],
        summary: ExtractionBenchmarkSummary | None,
        _predictions_path_tmp: Path,
    ) -> None:
        if mlflow and summary is not None:
            log_metric_mlflow(summary, out_dir=self.out_directory)

        logger.info(f"Metadata written to {self.metadata_path}")
        with open(self.metadata_path, "w", encoding="utf8") as file:
            json.dump(run_result.result.get_metadata_as_dict(), file, ensure_ascii=False, indent=2)

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
    on_file_done: Callable[[ExtractionResult, Path, Path], None] | None = None

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
            on_file_done=self.on_file_done,
            runname=spec.name,
        ).execute()
