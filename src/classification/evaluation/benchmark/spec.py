"""BenchmarkSpec class specification and respective parsing for classification."""

from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class BenchmarkSpec:
    """Specification of a classification benchmark run.

    Attributes:
        name: Human-readable benchmark name (used for folder/run naming).
        file_path: JSON file to classify (ground truth JSON or predictions JSON).
        file_subset_directory: Folder containing subset PDFs; used to filter by filenames.
        ground_truth_path: Optional ground truth JSON path if file_path is a predictions JSON.
    """

    name: str
    file_path: Path
    file_subset_directory: Path | None
    ground_truth_path: Path | None


def parse_benchmark_spec(spec: str) -> BenchmarkSpec:
    """Parse a benchmark spec.

    Supported formats:
      - "<name>:<file_path>:<file_subset_directory>"
      - "<name>:<file_path>:<file_subset_directory>:<ground_truth_path>"

    Note: Use an empty subset directory (or omit it by passing "none") if you truly want
    the whole JSON file (but in practice you usually want subset for benchmarks).
    """
    parts = [p.strip() for p in spec.split(":")]

    if len(parts) not in (3, 4):
        raise click.BadParameter(
            f"Invalid --benchmark '{spec}'. Expected "
            "'<name>:<file_path>:<file_subset_directory>' "
            "or '<name>:<file_path>:<file_subset_directory>:<ground_truth_path>'."
        )

    name, file_str, subset_str = parts[0], parts[1], parts[2]
    gt_str = parts[3] if len(parts) == 4 else None

    if not name:
        raise click.BadParameter(f"Invalid --benchmark '{spec}': name is empty.")

    file_path = Path(file_str)
    if not file_path.exists():
        raise click.BadParameter(f"Invalid --benchmark '{spec}': file_path does not exist: {file_path}")

    file_subset_directory: Path | None
    if subset_str.lower() in ("none", ""):
        file_subset_directory = None
    else:
        file_subset_directory = Path(subset_str)
        if not file_subset_directory.exists():
            raise click.BadParameter(
                f"Invalid --benchmark '{spec}': file_subset_directory does not exist: {file_subset_directory}"
            )

    ground_truth_path: Path | None = None
    if gt_str:
        ground_truth_path = Path(gt_str)
        if not ground_truth_path.exists():
            raise click.BadParameter(
                f"Invalid --benchmark '{spec}': ground_truth_path does not exist: {ground_truth_path}"
            )

    return BenchmarkSpec(
        name=name,
        file_path=file_path,
        file_subset_directory=file_subset_directory,
        ground_truth_path=ground_truth_path,
    )
