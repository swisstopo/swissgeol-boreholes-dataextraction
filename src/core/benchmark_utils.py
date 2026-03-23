"""Utility functions for benchmarking."""

from __future__ import annotations

import abc
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from pydantic import BaseModel

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def delete_temporary(pattern: Path) -> None:
    """Delete temporary files matching a glob pattern.

    Only files ending with '.tmp' are deleted.

    Args:
        pattern (Path): Glob pattern to match files (e.g., '/path/*.tmp' or '/path/**/*.tmp').
    """
    for file in glob(str(pattern)):
        if Path(file).suffix == ".tmp":
            os.remove(file)


def read_mlflow_runid(filename: str) -> str | None:
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


def write_mlflow_runid(filename: str, runid: str) -> None:
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


class BenchmarkSummary(BaseModel, ABC):
    """Shared base class for benchmark summaries."""

    ground_truth_path: str | None
    n_documents: int

    @abstractmethod
    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float]:
        """Return metrics in a flattened form for summaries/CSV output."""
        raise NotImplementedError


@dataclass
class Metrics(metaclass=abc.ABCMeta):
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
    @staticmethod
    def micro_average(metric_list: list[Metrics]) -> Metrics:
        """Converts a list of metrics to a metric.

        Args:
            metric_list (list): The list of metrics.

        Returns:
            Metrics: Combined metrics.
        """
        tp = sum([metric.tp for metric in metric_list])
        fp = sum([metric.fp for metric in metric_list])
        fn = sum([metric.fn for metric in metric_list])
        return Metrics(tp=tp, fp=fp, fn=fn)


def relative_after_common_root(paths: Sequence[Path]) -> list[str]:
    """Return relative paths after the longest common path prefix.

    If a path equals the common root (relative path == "."),
    return a meaningful tail instead of ".".

    Args:
        paths: Paths to process.

    Returns:
        Relative path strings after the common root.
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
