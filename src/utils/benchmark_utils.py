"""Utility functions for benchmarking."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from extraction.evaluation.benchmark.spec import BenchmarkSpec

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
