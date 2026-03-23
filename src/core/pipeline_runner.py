"""Shared pipeline execution utilities for runner modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from core.benchmark_utils import (
    finalize_pipeline_run,
    prepare_pipeline_temp_paths,
    start_or_resume_mlflow_run,
)
from core.mlflow_tracking import mlflow

PredictionT = TypeVar("PredictionT")
SummaryT = TypeVar("SummaryT")


@dataclass
class PipelineRunResult(Generic[PredictionT]):
    """Container for the shared pipeline executor.

    Attributes:
        predictions: Pipeline-specific prediction result.
        n_documents: Number of processed documents.
        copy_source: Optional path to copy to final predictions during finalization.
            If None, the shared executor falls back to the temporary predictions path.
    """

    predictions: PredictionT
    n_documents: int
    copy_source: Path | None = None


def execute_pipeline_run(
    *,
    predictions_path: Path,
    resume: bool,
    is_nested: bool,
    cleanup_mlflow_tmp: bool,
    setup_mlflow_run: Callable[[str | None], str] | None,
    run_predictions: Callable[[Path], PipelineRunResult[PredictionT]],
    evaluate_predictions: Callable[[PipelineRunResult[PredictionT]], SummaryT | None],
    after_evaluation: Callable[[PipelineRunResult[PredictionT], SummaryT | None, Path], None] | None = None,
    copy_predictions_to_final: bool = False,
) -> SummaryT | None:
    """Execute a shared single-pipeline lifecycle.

    The lifecycle is:
      1. Prepare temp paths
      2. Start/resume MLflow run if configured
      3. Run pipeline-specific prediction logic
      4. Log common metrics
      5. Run pipeline-specific evaluation
      6. Run optional pipeline-specific post-processing/logging
      7. Finalize temp files / predictions / MLflow

    Args:
        predictions_path: Final predictions path for the pipeline run.
        resume: Whether to resume temp files / MLflow run.
        is_nested: Whether this run is nested under a parent multi-benchmark run.
        cleanup_mlflow_tmp: Whether stale mlflow tmp files should be removed when not resuming.
        setup_mlflow_run: Callback to start/resume the pipeline-specific MLflow run.
        run_predictions: Callback that performs pipeline-specific prediction work.
            Receives the temp predictions path.
        evaluate_predictions: Callback that evaluates the pipeline-specific prediction result.
        after_evaluation: Optional callback for pipeline-specific logging/artifacts/final side effects.
            Receives the run result, summary, and temp predictions path.
        copy_predictions_to_final: Whether to copy predictions to final_predictions_path on finalize.

    Returns:
        SummaryT | None: Pipeline-specific summary.
    """
    temp_paths = prepare_pipeline_temp_paths(
        predictions_path,
        resume=resume,
        cleanup_mlflow_tmp=cleanup_mlflow_tmp,
    )
    predictions_path_tmp = temp_paths.predictions_path_tmp
    mlflow_runid_tmp = temp_paths.mlflow_runid_tmp

    if setup_mlflow_run is not None:
        start_or_resume_mlflow_run(
            resume=resume,
            mlflow_runid_tmp=mlflow_runid_tmp,
            setup_run=setup_mlflow_run,
        )

    run_result = run_predictions(predictions_path_tmp)

    if mlflow:
        mlflow.log_metric("n_documents", float(run_result.n_documents))

    summary = evaluate_predictions(run_result)

    if after_evaluation is not None:
        after_evaluation(run_result, summary, predictions_path_tmp)

    finalize_pipeline_run(
        is_nested=is_nested,
        predictions_path_tmp=run_result.copy_source or predictions_path_tmp,
        final_predictions_path=predictions_path,
        copy_predictions=copy_predictions_to_final,
        mlflow_runid_tmp=mlflow_runid_tmp,
    )

    return summary
