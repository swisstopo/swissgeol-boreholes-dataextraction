"""Utility functions for benchmarking."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from swissgeol_doc_processing.utils.file_utils import read_params

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")


def _relative_after_common_root(paths: Sequence[Path]) -> list[str]:
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


# def _parent_input_directory_key(benchmarks: Sequence[BenchmarkSpec]) -> str:
#     """Generate a parent input directory based on the input directories of the child runs.

#     Group_key is stable for the same set of child inputs and ground truths.

#     Args:
#         benchmarks (Sequence[BenchmarkSpec]): The list of benchmark specifications.

#     Return: str: group_key
#     """
#     # Sort to make it order-invariant
#     paths = [Path(b.input_path) for b in benchmarks]
#     inputs = " | ".join(sorted(_relative_after_common_root(paths)))

#     group_key = f"multi:{inputs}"
#     return group_key

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
