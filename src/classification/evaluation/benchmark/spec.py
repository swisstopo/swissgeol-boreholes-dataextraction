"""BenchmarkSpec class specification and respective parsing for classification."""

from dataclasses import dataclass
from pathlib import Path


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


def parse_benchmark_spec(value: str) -> BenchmarkSpec:
    """Parse a benchmark spec.

    Args:
        value: A string in one of the following formats:
            1) '<name>:<file_path>:<file_subset_directory>'
            2) '<name>:<file_path>:<file_subset_directory>:<ground_truth_path>'
            3) '<name>:<predictions_path>:<ground_truth_path>'

    Returns:
        A BenchmarkSpec instance with the parsed values.

    """
    # Split on ':' and trim whitespace to be tolerant of CLI input
    parts = [p.strip() for p in value.split(":")]

    # Case 1: Three-part specification
    if len(parts) == 3:
        name, a, b = parts
        a_path = Path(a)
        b_path = Path(b)

        # If both paths end in '.json', interpret this as
        #  "<name>:<predictions_path>:<ground_truth_path>"
        if a_path.suffix.lower() == ".json" and b_path.suffix.lower() == ".json":
            predictions_path = a_path
            ground_truth_path = b_path
            # In this mode, the subset directory is implicitly derived from the predictions file location
            subset_dir = predictions_path.parent
            return BenchmarkSpec(
                name=name,
                file_path=predictions_path,
                file_subset_directory=subset_dir,
                ground_truth_path=ground_truth_path,
            )
        # Otherwise interpret as:
        # "<name>:<file_path>:<subset_dir>"
        # (no explicit ground truth provided)
        return BenchmarkSpec(
            name=name,
            file_path=a_path,
            file_subset_directory=b_path,
            ground_truth_path=None,
        )

    # Case 2: Fully explicit four-part specification
    if len(parts) == 4:
        name, file_path, subset_dir, ground_truth = parts
        return BenchmarkSpec(
            name=name,
            file_path=Path(file_path),
            file_subset_directory=Path(subset_dir),
            ground_truth_path=Path(ground_truth),
        )
    # Any other number of parts is considered invalid input
    raise ValueError(
        f"Invalid --benchmark '{value}'. Expected "
        "'<name>:<file_path>:<file_subset_directory>' or "
        "'<name>:<file_path>:<file_subset_directory>:<ground_truth_path>' or "
        "'<name>:<predictions_path>:<ground_truth_path>'."
    )
