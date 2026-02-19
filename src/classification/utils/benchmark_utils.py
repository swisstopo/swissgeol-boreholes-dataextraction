"""Utility functions for benchmarking in the classification pipeline."""

import json
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


def count_documents(file_path: Path, file_subset_directory: Path | None) -> int:
    """Count number of documents.

    the function supports counting documents in the following scenarios:
        - file_subset_directory is provided and exists: count PDFs in the subset directory (priority)
        - file_path is a directory: count PDFs in the directory
        - file_path is a JSON file: count top-level keys (prediction mode)
        - file_path is a single PDF: count as 1 document
        - otherwise: return 0

    Args:
        file_path (Path): The main file path (could be a directory, a JSON file, or a single PDF).
        file_subset_directory (Path | None): An optional subset directory to count PDFs from

    Returns:
        int: The count of documents.
    """
    # 1) subset directory is prioritized if provided and valid
    if file_subset_directory and file_subset_directory.exists() and file_subset_directory.is_dir():
        subset_pdfs = list(file_subset_directory.glob("*.pdf"))
        if subset_pdfs:
            return len(subset_pdfs)

    if not file_path or not file_path.exists():
        return 0

    # 2) file_path as directory: count PDFs inside
    if file_path.is_dir():
        return len(list(file_path.glob("*.pdf")))

    # 3) file_path as a single file
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return 1

    if suffix == ".json":
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0

        # common format for prediction.json:
        # { "A1156.pdf": {...}, "3257.pdf": {...}, ... }
        if isinstance(data, dict):
            return len(data)

        return 0

    return 0
