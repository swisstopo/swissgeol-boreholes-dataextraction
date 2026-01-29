"""BenchmarkSpec class specification and respective parsing."""

from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class BenchmarkSpec:
    """Class for storing the specification of a benchmark for multiple runs."""

    name: str
    input_path: Path
    ground_truth_path: Path


def parse_benchmark_spec(spec: str) -> BenchmarkSpec:
    """Parse and validate a benchmark specification string.

    The expected format is:
        "<name>:<input_path>:<ground_truth_path>"

    Args:
        spec (str): Benchmark specification string passed via the CLI.

    Returns:
        BenchmarkSpec: Validated benchmark instance.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise click.BadParameter(f"Invalid --benchmark '{spec}'. Expected '<name>:<input_path>:<ground_truth_path>'.")

    name, input_str, gt_str = parts
    name = name.strip()
    input_path = Path(input_str.strip())
    ground_truth_path = Path(gt_str.strip())

    if not name:
        raise click.BadParameter(f"Invalid --benchmark '{spec}': name is empty.")
    if not input_path.exists():
        raise click.BadParameter(f"Invalid --benchmark '{spec}': input path does not exist: {input_path}")
    if not ground_truth_path.exists():
        raise click.BadParameter(
            f"Invalid --benchmark '{spec}': ground truth path does not exist: {ground_truth_path}"
        )

    return BenchmarkSpec(name=name, input_path=input_path, ground_truth_path=ground_truth_path)
