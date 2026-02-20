"""Utility functions for benchmarking in the extraction pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from core.mlflow_tracking import mlflow
from extraction.evaluation.benchmark.score import ExtractionBenchmarkSummary
from extraction.evaluation.benchmark.spec import BenchmarkSpec

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _relative_after_common_root_extraction(paths: Sequence[Path]) -> list[str]:
    """Return relative paths after the longest common path prefix.

    Args:
        paths (Sequence[Path]): The list of paths.

    Returns: list[str]: The list of relative paths.
    """
    resolved = [p.resolve() for p in paths]
    common_root = Path(os.path.commonpath(resolved))

    rels: list[str] = []
    for p in resolved:
        try:
            rels.append(str(p.relative_to(common_root)))
        except ValueError:
            # Defensive fallback (should not happen)
            rels.append(p.name)

    return rels


def _parent_input_directory_key_extraction(benchmarks: Sequence[BenchmarkSpec]) -> str:
    """Generate a parent input directory based on the input directories of the child runs.

    Group_key is stable for the same set of child inputs and ground truths.

    Args:
        benchmarks (Sequence[BenchmarkSpec]): The list of benchmark specifications.

    Return: str: group_key
    """
    # Sort to make it order-invariant
    paths = [Path(b.input_path) for b in benchmarks]
    inputs = " | ".join(sorted(_relative_after_common_root_extraction(paths)))

    group_key = f"multi:{inputs}"
    return group_key


def log_metric_mlflow(
    summary: ExtractionBenchmarkSummary,
    out_dir: Path,
    artifact_name: str = "benchmark_summary.json",
) -> None:
    """Log benchmark metrics + a JSON summary artifact to the *currently active* MLflow run.

    - Does NOT start/end MLflow runs.
    """
    metrics = summary.metrics(short=True)
    mlflow.log_metrics(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / artifact_name
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    mlflow.log_artifact(str(summary_path), artifact_path="summary")
