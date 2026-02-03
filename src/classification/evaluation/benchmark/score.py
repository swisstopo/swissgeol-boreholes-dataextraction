"""Score.py."""

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

    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float]:
        out: dict[str, float] = {}

        def add(path: str, obj: Any) -> None:
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
