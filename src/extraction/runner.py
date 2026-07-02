"""Pipeline runner for borehole data extraction with single and multi-benchmark support."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.ground_truth import GroundTruth
from core.pipeline_runner import MultiBenchmarkRunner, PipelineRunner, PipelineRunResult
from core.wandb_tracking import wandb, wandb_tracking
from extraction.core.extract import ExtractionResult, extract
from extraction.evaluation.benchmark.score import (
    ExtractionBenchmarkSummary,
    evaluate_all_predictions,
    evaluate_prediction,
)
from extraction.evaluation.benchmark.spec import BenchmarkSpec
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics, create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")

logger = logging.getLogger(__name__)


def _git_metadata() -> dict:
    try:
        import pygit2

        repo = pygit2.Repository(".")
        commit = repo[repo.head.target]
        return {
            "git_branch": repo.head.shorthand,
            "git_commit_sha": str(commit.id)[:8],
            "git_commit_message": commit.message.strip(),
        }
    except Exception:
        return {}


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
    wandb_group: str | None = None
    wandb_parent_run_id: str | None = None
    analytics: MatchingParamsAnalytics | None = field(init=False, default=None)
    _run_start_time: float = field(init=False, default=0.0)
    _logged_image_paths: list[Path] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.analytics = create_analytics() if self.options.matching_analytics else None
        self.copy_predictions_to_final = self.options.part == "all"
        self.out_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

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
        self._run_start_time = time.time()
        # Look for files to process
        pdf_files: list[Path] = (
            [self.input_directory] if self.input_directory.is_file() else list(self.input_directory.glob("*.pdf"))
        )
        n_documents = len(pdf_files)
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None

        # Load any partially-completed predictions for resume support
        predictions = read_json_predictions(predictions_path_tmp)

        if wandb_tracking and wandb is not None:
            self._init_wandb()

        for pdf_file in tqdm(pdf_files, desc="Processing files", unit="file"):
            # Check if file is already computed in previous run
            if predictions.contains(pdf_file.name):
                logger.info(f"{pdf_file.name} already predicted.")
                continue

            logger.info(f"Processing file: {pdf_file.name}")

            file_start_time = time.time()
            result = extract(file=pdf_file, filename=pdf_file.name, part=self.options.part, analytics=self.analytics)
            prediction_with_metrics = evaluate_prediction(result.predictions, ground_truth)
            predictions.add_file_predictions(prediction_with_metrics)

            if self.on_file_done is not None:
                # Pass updated results to callback
                self.on_file_done(result, self.out_directory, pdf_file)

            logger.info(f"Writing predictions to tmp JSON file {predictions_path_tmp}")
            write_json_predictions(path=predictions_path_tmp, predictions=predictions)

            if wandb_tracking and wandb is not None:
                self._log_file_artifacts_to_wandb(file_start_time)

        return PipelineRunResult(result=predictions, n_documents=n_documents)

    def evaluate(self, run_result: PipelineRunResult[OverallFilePredictions]) -> ExtractionBenchmarkSummary | None:
        ground_truth = GroundTruth(self.ground_truth_path) if self.ground_truth_path else None
        eval_summary = evaluate_all_predictions(
            predictions=run_result.result,
            ground_truth=ground_truth,
            out_directory=self.out_directory if wandb_tracking else None,
        )
        if eval_summary is not None:
            eval_summary.n_documents = run_result.n_documents
        return eval_summary

    def _init_wandb(self) -> None:
        config = {
            "input_directory": str(self.input_directory),
            "ground_truth_path": str(self.ground_truth_path) if self.ground_truth_path else None,
            **_git_metadata(),
            **flatten(line_detection_params),
            **flatten(matching_params),
        }
        if self.wandb_parent_run_id:
            config["parent_run_id"] = self.wandb_parent_run_id
            config["benchmark_id"] = self.wandb_group
            config["child_role"] = self.runname or "extraction"

        job_type = "benchmark-child" if self.wandb_parent_run_id else "extraction"
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "swissgeol-boreholes"),
            name=self.runname or "extraction",
            tags=["boreholes", "extraction"],
            group=self.wandb_group,
            job_type=job_type,
            config=config,
        )

    def _log_file_artifacts_to_wandb(self, file_start_time: float) -> None:
        draw_dir = self.out_directory / "draw"
        if draw_dir.exists():
            known = set(self._logged_image_paths)
            new_images = [p for p in sorted(draw_dir.rglob("*.png")) if p not in known]
            if new_images:
                self._logged_image_paths.extend(new_images)
                table = wandb.Table(columns=["filename", "image"])
                for img_path in self._logged_image_paths:
                    table.add_data(img_path.name, wandb.Image(str(img_path), caption=img_path.name))
                wandb.log({"png_browser": table})

        csv_dir = self.out_directory / "csv"
        if csv_dir.exists():
            new_csvs = [p for p in sorted(csv_dir.rglob("*.csv")) if p.stat().st_mtime >= file_start_time - 1]
            if new_csvs:
                wandb_csv_dir = Path(wandb.run.dir) / "media" / "csv"
                wandb_csv_dir.mkdir(parents=True, exist_ok=True)
                for csv_path in new_csvs:
                    dest = wandb_csv_dir / csv_path.name
                    shutil.copy(str(csv_path), str(dest))
                    wandb.save(str(dest), base_path=wandb.run.dir, policy="now")

    def after_evaluation(
        self,
        run_result: PipelineRunResult[OverallFilePredictions],
        summary: ExtractionBenchmarkSummary | None,
        _predictions_path_tmp: Path,
    ) -> None:
        logger.info(f"Metadata written to {self.metadata_path}")
        with open(self.metadata_path, "w", encoding="utf8") as file:
            json.dump(run_result.result.get_metadata_as_dict(), file, ensure_ascii=False, indent=2)

        if self.options.matching_analytics and self.analytics is not None:
            analytics_output_path = self.out_directory / "matching_params_analytics.json"
            self.analytics.save_analytics(analytics_output_path)
            logger.info(f"Matching parameters analytics saved to {analytics_output_path}")

        if self.options.part == "all":
            logger.info(f"Writing predictions to final JSON file {self.predictions_path}")

        if wandb_tracking and wandb is not None:
            self._log_to_wandb(run_result, summary)

    def _log_to_wandb(
        self,
        run_result: PipelineRunResult[OverallFilePredictions],
        summary: ExtractionBenchmarkSummary | None,
    ) -> None:
        # wandb was already initialized in run_predictions(); if for some reason it wasn't
        # (e.g. run_predictions skipped), init it now as a fallback.
        if wandb.run is None:
            self._init_wandb()
        try:
            base_metrics = {"n_documents": float(run_result.n_documents)}
            eval_metrics = summary.metrics_flat() if summary else {}
            all_metrics = {k: v for k, v in {**base_metrics, **eval_metrics}.items() if v is not None}
            wandb.log(all_metrics)
            wandb.run.summary.update(all_metrics)

            if summary:
                summary_path = self.out_directory / "benchmark_summary.json"
                with open(summary_path, "w", encoding="utf8") as f:
                    json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)
                wandb.save(str(summary_path), base_path=str(self.out_directory), policy="now")
                for csv_name in ("document_level_metadata_metrics.csv", "document_level_geology_metrics.csv"):
                    csv_path = self.out_directory / csv_name
                    if csv_path.exists():
                        wandb.save(str(csv_path), base_path=str(self.out_directory), policy="now")
        finally:
            wandb.finish()


@dataclass(kw_only=True)
class ExtractionBenchmarkRunner(MultiBenchmarkRunner[BenchmarkSpec, ExtractionBenchmarkSummary]):
    """Orchestrates multiple extraction benchmarks."""

    experiment_name = "Boreholes data extraction"
    input_tag_name = "input_directory"
    input_path_attr = "input_path"
    aggregate_label = "overall"
    runname = "benchmark"

    options: ExtractionOptions = field(default_factory=ExtractionOptions)
    on_file_done: Callable[[ExtractionResult, Path, Path], None] | None = None
    _wandb_group: str | None = field(init=False, default=None)
    _wandb_parent_run_id: str | None = field(init=False, default=None)

    def _init_wandb_parent(self) -> None:
        import datetime

        self._wandb_group = f"benchmark-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        parent_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "swissgeol-boreholes"),
            name="benchmark-parent",
            group=self._wandb_group,
            job_type="orchestrator",
            tags=["boreholes", "benchmark", "parent"],
            config={
                "benchmarks": [spec.name for spec in self.benchmarks],
                "n_benchmarks": len(self.benchmarks),
                **_git_metadata(),
            },
        )
        self._wandb_parent_run_id = parent_run.id
        wandb.finish()

    def run_single(self, spec: BenchmarkSpec) -> ExtractionBenchmarkSummary | None:
        logger.info("Running benchmark: %s", spec.name)
        if self._wandb_group is None and wandb_tracking:
            self._init_wandb_parent()

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
            wandb_group=self._wandb_group,
            wandb_parent_run_id=self._wandb_parent_run_id,
        ).execute()

    def finalize_summary(
        self, overall_results: list[tuple[str, ExtractionBenchmarkSummary | None]], root: Path
    ) -> None:
        """Write overall_summary.csv and log aggregate metrics to W&B."""
        super().finalize_summary(overall_results, root)

        if not (wandb_tracking and wandb is not None and self._wandb_group):
            return

        summary_csv_path = root / "overall_summary.csv"
        if not summary_csv_path.exists():
            return

        df = pd.read_csv(summary_csv_path)
        means = (
            df.drop(columns=["benchmark", "ground_truth_path"], errors="ignore")
            .mean(numeric_only=True)
            .round(3)
            .to_dict()
        )
        aggregate = {
            "n_benchmarks": len(overall_results),
            "total_documents": int(df["n_documents"].sum()) if "n_documents" in df.columns else 0,
            **means,
        }

        # Resume the parent orchestrator run to log aggregate metrics alongside initial config.
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "swissgeol-boreholes"),
            id=self._wandb_parent_run_id,
            resume="allow",
            group=self._wandb_group,
        )
        try:
            wandb.log(aggregate)
            wandb.run.summary.update(aggregate)
            wandb.save(str(summary_csv_path), policy="now")
        finally:
            wandb.finish()
