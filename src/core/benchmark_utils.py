"""Utility functions for benchmarking."""

from __future__ import annotations

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from core.mlflow_tracking import mlflow

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def delete_temporary(pattern: Path | str) -> None:
    """Delete temporary files matching a glob pattern.

    Only files ending with '.tmp' are deleted.

    Args:
        pattern (Path): Glob pattern to match files (e.g., '/path/*.tmp' or '/path/**/*.tmp').
    """
    for file in glob(str(pattern)):
        if Path(file).suffix == ".tmp":
            os.remove(file)


def read_mlflow_runid(filename: Path | str) -> str | None:
    """Read locally stored mlflow run id.

    Args:
        filename (str): Name of the file that contains runid.

    Returns:
        str | None: Loaded runid if any, otherwise None.
    """
    if not Path(filename).exists():
        return None

    with open(filename, encoding="utf8") as f:
        return json.load(f)


def write_mlflow_runid(filename: Path | str, runid: str) -> None:
    """Locally stores mlflow run id.

    Args:
        filename (str): Name of the file to store runid.
        runid (str): Runid to store.
    """
    with open(filename, "w", encoding="utf8") as file:
        json.dump(runid, file, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the application."""
    logging.basicConfig(format=DEFAULT_FORMAT, level=level, datefmt=DEFAULT_DATEFMT)


def short_metric_key(k: str) -> str:
    """Convert nested metric keys to the short metric key used in child runs.

    Examples:
      metrics/geology/layer_f1 -> layer_f1
      metrics/metadata/name_f1 -> name_f1
      geology/layer_f1 -> layer_f1
      metadata/name_f1 -> name_f1
      layer_f1 -> layer_f1
    """
    if k.startswith("metrics/"):
        k = k[len("metrics/") :]
    return k.split("/", 1)[1] if "/" in k else k


class BenchmarkSummary(BaseModel, ABC):
    """Shared base class for benchmark summaries."""

    ground_truth_path: str | None
    n_documents: int

    @abstractmethod
    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float]:
        """Return metrics in a flattened form for summaries/CSV output."""
        raise NotImplementedError


@dataclass(frozen=True)
class Metrics:
    """Metrics for the evaluation of extracted features (e.g., Groundwater, Elevation, Coordinates)."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """Calculates the precision.

        Returns:
            float: The precision.
        """
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0

    @property
    def recall(self) -> float:
        """Calculates the recall.

        Returns:
            float: The recall.
        """
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0

    @property
    def f1(self) -> float:
        """Calculates the F1 score.

        Returns:
            float: The F1 score.
        """
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    def to_json(self, feature_name) -> dict[str, float]:
        """Converts the object to a dictionary.

        Args:
            feature_name (str): Name of the feature to append.

        Returns:
            dict[str, float]: The object as a dictionary.
        """
        return {
            f"{feature_name}_precision": self.precision,
            f"{feature_name}_recall": self.recall,
            f"{feature_name}_f1": self.f1,
        }

    # TODO: Currently, some other methods for averaging metrics are in the OverallMetrics class.
    # On the long run, we should refactor this to have a single place where these averaging computations are
    # implemented.
    @classmethod
    def micro_average(cls, metric_list: list[Metrics]) -> Metrics:
        """Converts a list of metrics to a metric.

        Args:
            metric_list (list): The list of metrics.

        Returns:
            Metrics: Combined metrics with the same type as the caller.
        """
        tp = sum(metric.tp for metric in metric_list)
        fp = sum(metric.fp for metric in metric_list)
        fn = sum(metric.fn for metric in metric_list)
        return cls(tp=tp, fp=fp, fn=fn)


def relative_after_common_root(paths: Sequence[Path]) -> list[str]:
    """Return relative paths after the longest common path prefix.

    If a path equals the common root (relative path == "."),
    return a meaningful tail instead of ".".

    Args:
        paths: Paths to process.

    Returns:
        list[str]: Relative path strings after the common root.
    """
    if not paths:
        return []

    resolved = [p.expanduser().resolve() for p in paths]

    try:
        common_root = Path(os.path.commonpath([str(p) for p in resolved]))
    except Exception:
        return [p.name for p in resolved]

    rels: list[str] = []
    for p in resolved:
        try:
            rel = p.relative_to(common_root)
            if rel == Path("."):
                rels.append(str(Path(*p.parts[-2:])))  # last 2 parts
            else:
                rels.append(str(rel))
        except ValueError:
            rels.append(p.name)

    return rels


def parent_input_key(paths: Sequence[Path]) -> str:
    """Generate a stable parent input key for a group of child inputs.

    Args:
        paths: Child input paths.

    Returns:
        Group key string.
    """
    inputs = " | ".join(sorted(relative_after_common_root(paths)))
    return f"multi:{inputs}"


def finalize_overall_summary(
    *,
    overall_results: list[tuple[str, Any | None]],
    multi_root: Path,
    aggregate_label: str,
    metric_key_shortener,
) -> None:
    """Write overall_summary.csv and optionally log aggregate metrics to MLflow.

    Args:
        overall_results: List of tuples (benchmark_name, summary). Each summary is expected
            to expose:
              - ground_truth_path
              - n_documents
              - metrics_flat()
        multi_root: Root output directory where overall_summary.csv is written.
        aggregate_label: Label for the aggregate row, e.g. "overall" or "total/mean".
        metric_key_shortener: Function that converts full metric keys to shorter MLflow keys.
    """
    summary_csv_path = multi_root / "overall_summary.csv"

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
    agg_row.at["benchmark"] = aggregate_label
    agg_row.at["n_documents"] = total_docs
    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    df.to_csv(summary_csv_path, index=False)

    if mlflow:
        for full_key, value in means.items():
            if pd.notna(value):
                short_key = metric_key_shortener(full_key)
                mlflow.log_metric(short_key, value)

        if pd.notna(total_docs):
            mlflow.log_metric("n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")


@dataclass(frozen=True)
class PipelineTempPaths:
    """Temporary file paths used during a single pipeline run."""

    predictions_path_tmp: Path
    mlflow_runid_tmp: Path


def prepare_pipeline_temp_paths(
    predictions_path: Path,
    resume: bool,
    cleanup_mlflow_tmp: bool = True,
) -> PipelineTempPaths:
    """Prepare temporary file paths for a pipeline run.

    Creates the parent directory for predictions_path and derives:
    - <predictions_path>.tmp
    - mlflow_runid.json.tmp in the same directory

    If resume is False, stale temporary files are removed first.

    Args:
        predictions_path (Path): Final predictions path for the run.
        resume (bool): Whether the run should resume from previous temp files.
        cleanup_mlflow_tmp (bool): Whether to remove the mlflow temp file when not resuming.

    Returns:
        PipelineTempPaths: The derived temporary paths.
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_path_tmp = predictions_path.parent / f"{predictions_path.name}.tmp"
    mlflow_runid_tmp = predictions_path.parent / "mlflow_runid.json.tmp"

    if not resume:
        delete_temporary(predictions_path_tmp)
        if cleanup_mlflow_tmp:
            delete_temporary(mlflow_runid_tmp)

    return PipelineTempPaths(
        predictions_path_tmp=predictions_path_tmp,
        mlflow_runid_tmp=mlflow_runid_tmp,
    )


def finalize_pipeline_run(
    is_nested: bool,
    predictions_path_tmp: Path | None = None,
    final_predictions_path: Path | None = None,
    copy_predictions: bool = False,
    mlflow_runid_tmp: Path | None = None,
) -> None:
    """Finalize a pipeline run.

    Optionally copies temp predictions to the final path, removes temporary files
    for non-nested runs, and ends the MLflow run if tracking is enabled.

    Args:
        is_nested (bool): Whether this run is nested under a parent benchmark run.
        predictions_path_tmp (Path | None): Temp predictions path.
        final_predictions_path (Path | None): Final predictions path.
        copy_predictions (bool): Whether to copy temp predictions to final path.
        mlflow_runid_tmp (Path | None): Temp MLflow run id path.
    """
    if (
        copy_predictions
        and predictions_path_tmp is not None
        and final_predictions_path is not None
        and predictions_path_tmp.exists()
    ):
        shutil.copy(src=predictions_path_tmp, dst=final_predictions_path)

    if not is_nested:
        if predictions_path_tmp is not None:
            delete_temporary(predictions_path_tmp)
        if mlflow_runid_tmp is not None:
            delete_temporary(mlflow_runid_tmp)

    if mlflow:
        mlflow.end_run()
