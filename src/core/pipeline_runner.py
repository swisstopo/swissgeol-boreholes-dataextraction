"""Abstract pipeline runner classes for single and multi-benchmark execution."""

from __future__ import annotations

import abc
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import pandas as pd

from core.benchmark_utils import (
    BenchmarkSummary,
    delete_temporary,
    parent_input_key,
    prepare_pipeline_temp_paths,
    read_mlflow_runid,
    short_metric_key,
    write_mlflow_runid,
)
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_tracking

PredictionT = TypeVar("PredictionT")
SummaryT = TypeVar("SummaryT", bound=BenchmarkSummary)


class HasName(Protocol):
    """Protocol for benchmark specs that expose a name."""

    name: str


SpecT = TypeVar("SpecT", bound=HasName)


@dataclass
class PipelineRunResult(Generic[PredictionT]):
    """Container for pipeline prediction output.

    Attributes:
        result: Pipeline-specific prediction result.
        n_documents: Number of processed documents.
    """

    result: PredictionT
    n_documents: int


@dataclass(kw_only=True)
class PipelineRunner(abc.ABC, Generic[PredictionT, SummaryT]):
    """Abstract base for pipeline execution with MLflow tracking and resumable runs.

    Subclasses implement prediction, evaluation, and optional post-evaluation logic.
    The shared lifecycle (temp paths, MLflow, finalization) is handled by execute().
    """

    predictions_path: Path
    resume: bool = False
    is_nested: bool = False
    cleanup_mlflow_tmp: bool = True
    copy_predictions_to_final: bool = False

    def __post_init__(self) -> None:
        if mlflow and type(self).setup_mlflow_run is PipelineRunner.setup_mlflow_run:
            raise NotImplementedError(
                f"{type(self).__name__} must implement setup_mlflow_run when MLflow tracking is active"
            )

    @abc.abstractmethod
    def run_predictions(self, predictions_path_tmp: Path) -> PipelineRunResult[PredictionT]:
        """Execute pipeline-specific prediction logic."""

    @abc.abstractmethod
    def evaluate(self, run_result: PipelineRunResult[PredictionT]) -> SummaryT | None:
        """Evaluate predictions and return a benchmark summary."""

    def setup_mlflow_run(self, runid: str | None) -> str:
        """Start or resume an MLflow run. Must be overridden when MLflow tracking is active.

        Args:
            runid: Previous run id to resume, or None to start a new run.

        Returns:
            The active MLflow run id.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement setup_mlflow_run")

    def after_evaluation(
        self,
        run_result: PipelineRunResult[PredictionT],
        summary: SummaryT | None,
        predictions_path_tmp: Path,
    ) -> None:
        """Optional post-evaluation hook for logging, artifacts, or side effects."""

    def execute(self) -> SummaryT | None:
        """Run the complete pipeline lifecycle.

        Lifecycle:
          1. Prepare temp paths
          2. Start/resume MLflow run (if tracking is enabled)
          3. Run predictions
          4. Log n_documents metric
          5. Evaluate predictions
          6. Run post-evaluation hook
          7. Finalize temp files and MLflow run

        Returns:
            Pipeline-specific benchmark summary, or None if evaluation is skipped.
        """
        temp_paths = prepare_pipeline_temp_paths(self.predictions_path, self.resume, self.cleanup_mlflow_tmp)

        if mlflow:
            runid = read_mlflow_runid(temp_paths.mlflow_runid_tmp) if self.resume else None
            runid = self.setup_mlflow_run(runid)
            write_mlflow_runid(temp_paths.mlflow_runid_tmp, runid)

        run_result = self.run_predictions(temp_paths.predictions_path_tmp)

        if mlflow:
            mlflow.log_metric("n_documents", float(run_result.n_documents))

        summary = self.evaluate(run_result)
        self.after_evaluation(run_result, summary, temp_paths.predictions_path_tmp)

        self._finalize_run(temp_paths.predictions_path_tmp, temp_paths.mlflow_runid_tmp)

        return summary

    def _finalize_run(self, predictions_path_tmp: Path, mlflow_runid_tmp: Path | None = None) -> None:
        """Finalize a pipeline run.

        Optionally copies temp predictions to the final path, removes temporary files
        for non-nested runs, and ends the MLflow run if tracking is enabled.

        Args:
        predictions_path_tmp: Temporary path where predictions were written.
        mlflow_runid_tmp: Temporary path where MLflow run ID is stored, if applicable.
        """
        if self.copy_predictions_to_final and predictions_path_tmp.exists():
            shutil.copy(src=predictions_path_tmp, dst=self.predictions_path)

        if not self.is_nested:
            delete_temporary(predictions_path_tmp)
            if mlflow_runid_tmp is not None:
                delete_temporary(mlflow_runid_tmp)

        if mlflow:
            mlflow.end_run()


@dataclass(kw_only=True)
class MultiBenchmarkRunner(abc.ABC, Generic[SpecT, SummaryT]):
    """Abstract base for orchestrating multiple benchmarks with shared MLflow tracking.

    Subclasses implement single-benchmark execution, parent MLflow setup, and summary finalization.
    The shared orchestration (directory creation, MLflow parent run, cleanup) is handled by run().
    """

    benchmarks: Sequence[SpecT]
    multi_root: Path
    resume: bool

    experiment_name: ClassVar[str]
    input_tag_name: ClassVar[str]
    input_path_attr: ClassVar[str]
    aggregate_label: ClassVar[str]
    runname: ClassVar[str | None] = None
    _mlflow_use_parent_out_dir: ClassVar[bool] = False

    @property
    def _parent_runid_tmp(self) -> Path:
        return self.multi_root / "mlflow_parent_runid.json.tmp"

    @abc.abstractmethod
    def run_single(self, spec: SpecT) -> SummaryT | None:
        """Execute a single benchmark and return its summary."""

    def finalize_summary(self, overall_results: list[tuple[str, SummaryT | None]], root: Path) -> None:
        """Write overall_summary.csv and optionally log aggregate metrics to MLflow.

        Args:
        overall_results: List of tuples (benchmark_name, summary). Each summary is expected
            to expose:
              - ground_truth_path
              - n_documents
              - metrics_flat()
        root: Directory where overall_summary.csv will be written.
        """
        summary_csv_path = root / "overall_summary.csv"

        rows = []
        for benchmark, summary in overall_results:
            row: dict[str, Any] = {"benchmark": benchmark}
            if summary is not None:
                row["ground_truth_path"] = summary.ground_truth_path
                row["n_documents"] = summary.n_documents
                row.update(summary.metrics_flat())
            rows.append(row)

        df = pd.DataFrame(rows).sort_values(by="benchmark")

        total_docs = df["n_documents"].sum()
        means = df.drop(columns=["n_documents"], errors="ignore").mean(numeric_only=True)

        agg_row = means.round(3)
        agg_row.at["benchmark"] = self.aggregate_label
        agg_row.at["n_documents"] = total_docs
        df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

        df.to_csv(summary_csv_path, index=False)

        if mlflow:
            for full_key, value in means.items():
                if pd.notna(value):
                    mlflow.log_metric(short_metric_key(full_key), value)

            if pd.notna(total_docs):
                mlflow.log_metric("n_documents", float(total_docs))

            mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")

    def run(self) -> None:
        """Execute all benchmarks and finalize the aggregate summary.

        Lifecycle:
          1. Create multi_root directory
          2. Start/resume MLflow parent run (if tracking is enabled)
          3. Run each benchmark via run_single()
          4. Finalize the aggregate summary
          5. End MLflow parent run and clean up temp files
        """
        self.multi_root.mkdir(parents=True, exist_ok=True)

        if mlflow:
            runid = read_mlflow_runid(self._parent_runid_tmp) if self.resume else None
            runid = self.setup_parent_run(runid)
            write_mlflow_runid(self._parent_runid_tmp, runid)

        overall_results: list[tuple[str, SummaryT | None]] = [
            (spec.name, self.run_single(spec)) for spec in self.benchmarks
        ]

        self.finalize_summary(overall_results, self.multi_root)

        if mlflow:
            mlflow.end_run()

        delete_temporary(self._parent_runid_tmp)
        delete_temporary(self.multi_root / "*" / "*.tmp")

    def setup_parent_run(self, runid: str | None) -> str:
        """Start or resume the shared MLflow parent run for multi-benchmark execution.

        Args:
        runid: Existing run ID to resume, or None.

        Returns: Active parent run ID.
        """
        tags = {
            self.input_tag_name: parent_input_key([Path(getattr(b, self.input_path_attr)) for b in self.benchmarks]),
            "out_directory": self.multi_root.parent if self._mlflow_use_parent_out_dir else None,
            "run_type": "multi_benchmark",
            "benchmarks": ",".join(b.name for b in self.benchmarks),
        }
        return setup_mlflow_tracking(
            run_id=runid,
            experiment_name=self.experiment_name,
            runname=self.runname,
            nested=False,
            tags=tags,
        )
