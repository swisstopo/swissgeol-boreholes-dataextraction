"""Score.py."""

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
    metrics: dict[str, float]

    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float]:
        out = {}
        for k, v in self.metrics.items():
            key = k if short else f"{prefix}/{k}"
            out[key] = float(v)
        return out
