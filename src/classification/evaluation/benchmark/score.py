"""Evaluate the predictions against the ground truth."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


class ClassificationBenchmarkSummary(BaseModel):
    """Helper class containing a summary of all the results of a single classification benchmark."""

    file_path: str
    ground_truth_path: str | None
    file_subset_directory: str | None
    n_layers: int
    classifier_type: str
    model_path: str | None
    classification_system: str
    metrics: dict[str, Any]

    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float] | None:
        """Flatten the metrics dictionary to a single level.

        Args:
            prefix (str): The prefix to use for the flattened keys.
            short (bool): Whether to use short keys (i.e. without the prefix).

        Returns:
            dict[str, float]: The flattened metrics dictionary.
        """
        out: dict[str, float] = {}

        def add(path: str, obj: Any) -> None:
            """Recursively add flattened metrics to the output dictionary.

            Args:
                path (str): The current path in the metrics hierarchy.
                obj (Any): The current metrics object to process.
            """
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    add(f"{path}/{k}" if path else str(k), v)
                return

            # skip Nones/bools and non-numerics
            if obj is None or isinstance(obj, bool):
                return
            try:
                out[path] = float(obj)
            except (TypeError, ValueError):
                return

        # flatten each top-level metric key
        for k, v in (self.metrics or {}).items():
            key = str(k) if short else f"{prefix}/{k}"
            add(key, v)

        return out
