"""Utility functions for benchmarking."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from extraction.evaluation.benchmark.score import BenchmarkSummary
from extraction.evaluation.benchmark.spec import BenchmarkSpec
from swissgeol_doc_processing.utils.file_utils import read_params

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")


def _relative_after_common_root(paths: Sequence[Path]) -> list[str]:
    """Return relative paths after the longest common path prefix.

    Guard: if a path equals the common root (relative path == "."),
    return a meaningful tail (e.g. filename) instead of ".".

    Args:
        paths (Sequence[Path]): The list of paths to process.

    Returns:
        list[str]: The list of relative paths after the common root.
    """
    if not paths:
        return []

    resolved = [p.expanduser().resolve() for p in paths]

    # If paths are on different roots/drives
    try:
        common_root = Path(os.path.commonpath([str(p) for p in resolved]))
    except Exception:
        # Fallback: just use names
        return [p.name for p in resolved]

    rels: list[str] = []
    for p in resolved:
        rel = p.relative_to(common_root)
        # avoid "." which makes the parent key useless
        if rel == Path("."):
            rels.append(str(Path(*p.parts[-2:])))  # last 2 parts
        else:
            rels.append(str(rel))

    return rels


T = TypeVar("T")


def parent_group_key(
    benchmarks: Sequence[T],
    get_path: Callable[[T], Path],
    *,
    prefix: str = "multi",
) -> str:
    """Stable group key for the benchmark set, based on a per-benchmark path."""
    paths = [get_path(b) for b in benchmarks]
    inputs = " | ".join(sorted(_relative_after_common_root(paths)))
    return f"{prefix}:{inputs}"


def _parent_input_directory_key(benchmarks: Sequence[BenchmarkSpec]) -> str:
    """Generate a parent input directory based on the input directories of the child runs.

    Args:
        benchmarks (Sequence[BenchmarkSpec]): The list of benchmark specifications.

    Return: str: group_key
    """
    paths: list[Path] = []

    for b in benchmarks:
        # extraction style
        if hasattr(b, "input_path") and b.input_path is not None:
            paths.append(Path(b.input_path))
            continue

        # classification style
        if hasattr(b, "file_path") and b.file_path is not None:
            paths.append(Path(b.file_path))
            continue

        raise AttributeError(f"BenchmarkSpec-like object {type(b)!r} has neither 'input_path' nor 'file_path'.")
    inputs = " | ".join(sorted(_relative_after_common_root(paths)))

    group_key = f"multi:{inputs}"
    return group_key


def _short_metric_key(k: str) -> str:
    """Drop the first namespace segment.

    Examples:
      geology/layer_f1 -> layer_f1
      metadata/name_f1 -> name_f1
      layer_f1 -> layer_f1

    Args:
        k (str): The original metric key.

    Returns:
        str: The shortened metric key.
    """
    return k.split("/", 1)[1] if "/" in k else k


def log_metric_mlflow(
    summary: BenchmarkSummary,
    out_dir: Path,
    artifact_name: str = "benchmark_summary.json",
) -> None:
    """Log benchmark metrics + a JSON summary artifact to the *currently active* MLflow run.

    - Does NOT start/end MLflow runs.
    """
    import mlflow

    metrics = summary.metrics(short=True)
    mlflow.log_metrics(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / artifact_name
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    mlflow.log_artifact(str(summary_path), artifact_path="summary")
