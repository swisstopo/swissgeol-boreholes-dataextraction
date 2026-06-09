"""BenchmarkSpec class specification and respective parsing for classification."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSpec:
    """Specification of a classification benchmark run.

    Attributes:
        name: Human-readable benchmark name (used for folder/run naming).
        file_path: Input path to classify.
    """

    name: str
    file_path: Path


def parse_benchmark_spec(value: str) -> BenchmarkSpec:
    """Parse a benchmark spec.

    Args:
        value: A string in the format:
            '<name>:<input_path>'

    Returns:
        A BenchmarkSpec instance with the parsed values.
    """
    # Split on ':' and trim whitespace to be tolerant of CLI input
    parts = [p.strip() for p in value.split(":")]

    if len(parts) != 2:
        raise ValueError(f"Invalid --benchmark '{value}'. Expected '<name>:<input_path>'.")

    name, file_path = parts
    return BenchmarkSpec(
        name=name,
        file_path=Path(file_path),
    )
