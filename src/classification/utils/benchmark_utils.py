"""Utility functions for benchmarking in the classification pipeline."""

import os
from collections.abc import Sequence
from pathlib import Path


def _relative_after_common_root_classification(paths: Sequence[Path]) -> list[str]:
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


def _parent_input_directory_key_classification(benchmarks: list[Path]) -> str:
    """Generate a parent input directory based on the input directories of the child runs.

    Args:
        benchmarks (list): The list of benchmark specifications.

    Return: str: group_key
    """
    inputs = " | ".join(sorted(_relative_after_common_root_classification(benchmarks)))
    group_key = f"multi:{inputs}"
    return group_key
