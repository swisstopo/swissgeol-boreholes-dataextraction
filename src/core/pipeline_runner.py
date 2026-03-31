"""Abstract pipeline runner classes for single and multi-benchmark execution."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from core.benchmark_utils import (
    BenchmarkSummary,
    delete_temporary,
    finalize_pipeline_run,
    prepare_pipeline_temp_paths,
    read_mlflow_runid,
    write_mlflow_runid,
)
from core.mlflow_tracking import mlflow

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

        finalize_pipeline_run(
            is_nested=self.is_nested,
            predictions_path_tmp=temp_paths.predictions_path_tmp,
            final_predictions_path=self.predictions_path,
            copy_predictions=self.copy_predictions_to_final,
            mlflow_runid_tmp=temp_paths.mlflow_runid_tmp,
        )

        return summary


@dataclass(kw_only=True)
class MultiBenchmarkRunner(abc.ABC, Generic[SpecT, SummaryT]):
    """Abstract base for orchestrating multiple benchmarks with shared MLflow tracking.

    Subclasses implement single-benchmark execution, parent MLflow setup, and summary finalization.
    The shared orchestration (directory creation, MLflow parent run, cleanup) is handled by run().
    """

    benchmarks: Sequence[SpecT]
    multi_root: Path
    resume: bool

    def __post_init__(self) -> None:
        if mlflow and type(self).setup_parent_run is MultiBenchmarkRunner.setup_parent_run:
            raise NotImplementedError(
                f"{type(self).__name__} must implement setup_parent_run when MLflow tracking is active"
            )

    @property
    def _parent_runid_tmp(self) -> Path:
        return self.multi_root / "mlflow_parent_runid.json.tmp"

    @abc.abstractmethod
    def run_single(self, spec: SpecT) -> SummaryT | None:
        """Execute a single benchmark and return its summary."""

    @abc.abstractmethod
    def finalize_summary(self, overall_results: list[tuple[str, SummaryT | None]], root: Path) -> None:
        """Write and log the aggregated summary across all benchmarks."""

    def setup_parent_run(self, runid: str | None) -> str:
        """Start or resume the MLflow parent run. Must be overridden when MLflow tracking is active.

        Args:
            runid: Previous parent run id to resume, or None to start a new run.

        Returns:
            The active MLflow parent run id.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement setup_parent_run")

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
