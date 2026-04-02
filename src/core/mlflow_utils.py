"""Shared MLflow utilities for extraction and classification pipelines."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.mlflow_tracking import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow_tracking(
    *,
    run_id: str | None,
    experiment_name: str,
    runname: str | None = None,
    nested: bool = False,
    tags: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Initialize and configure an MLflow run.

    Args:
        run_id: Existing run ID to resume, or None to start a new run.
        experiment_name: MLflow experiment name.
        runname: Optional MLflow run name.
        nested: Whether the run is nested.
        tags: Tags to set on the run. Values of None are ignored.
        params: Params to log. Values of None are ignored.

    Returns:
        str: The active MLflow run ID.
    """
    if not mlflow:
        raise ValueError("Tracking is not activated")

    mlflow.set_experiment(experiment_name)

    try:
        mlflow.start_run(run_name=runname, run_id=run_id, nested=nested)
    except mlflow.MlflowException:
        mlflow.start_run(run_name=runname, nested=nested)
        logger.warning(f"Unable to resume run with ID: {run_id} ({runname}), start new one.")

    for key, value in (tags or {}).items():
        if value is not None:
            mlflow.set_tag(key, str(value))

    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    if clean_params:
        mlflow.log_params(clean_params)

    import pygit2

    repo = pygit2.Repository(".")
    commit = repo[repo.head.target]
    mlflow.set_tag("git_branch", repo.head.shorthand)
    mlflow.set_tag("git_commit_message", commit.message)
    mlflow.set_tag("git_commit_sha", str(commit.id))

    return mlflow.active_run().info.run_id


def setup_mlflow_parent_run(
    *,
    run_id: str | None,
    experiment_name: str,
    parent_input_key: str,
    benchmarks: Sequence[Any],
    runname: str | None = None,
    input_tag_name: str = "input_path",
    out_directory: Path | str | None = None,
) -> str:
    """Start a shared parent MLflow run for multi-benchmark execution.

    Args:
        run_id: Existing run ID to resume, or None.
        experiment_name: MLflow experiment name.
        parent_input_key: Aggregate input identifier for the parent run.
        benchmarks: Benchmark specs with a `.name` attribute.
        runname: Optional run name.
        input_tag_name: Tag name used for the input identifier.
        out_directory: Optional output directory tag.

    Returns:
        str: Active parent run ID.
    """
    tags = {
        input_tag_name: parent_input_key,
        "out_directory": out_directory,
        "run_type": "multi_benchmark",
        "benchmarks": ",".join(b.name for b in benchmarks),
    }

    return setup_mlflow_tracking(
        run_id=run_id,
        experiment_name=experiment_name,
        runname=runname,
        nested=False,
        tags=tags,
    )
