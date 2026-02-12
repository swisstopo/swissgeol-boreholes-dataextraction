"""Utils for extraction pipeline and benchmarks."""

import json
import os
from pathlib import Path

from extraction.evaluation.benchmark.score import BenchmarkSummary

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow


def log_metric_mlflow(
    summary: BenchmarkSummary,
    out_dir: Path,
    artifact_name: str = "benchmark_summary.json",
) -> None:
    """Log benchmark metrics + a JSON summary artifact to the *currently active* MLflow run.

    - Does NOT start/end MLflow runs.
    """
    metrics = summary.metrics(short=True)
    mlflow.log_metrics(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / artifact_name
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    mlflow.log_artifact(str(summary_path), artifact_path="summary")
