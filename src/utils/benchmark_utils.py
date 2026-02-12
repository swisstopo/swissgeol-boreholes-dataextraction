"""Utility functions for benchmarking."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


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


def _parent_input_directory_key(benchmarks: list[Path]) -> str:
    """Generate a parent input directory based on the input directories of the child runs.

    Args:
        benchmarks (list): The list of benchmark specifications.

    Return: str: group_key
    """
    inputs = " | ".join(sorted(_relative_after_common_root(benchmarks)))
    group_key = f"multi:{inputs}"
    return group_key


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
