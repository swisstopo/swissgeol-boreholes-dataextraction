"""Utility functions for benchmarking."""

from __future__ import annotations

import json
import logging
import os
from glob import glob
from pathlib import Path

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
