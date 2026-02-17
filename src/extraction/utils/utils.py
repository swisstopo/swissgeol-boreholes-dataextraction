"""Utils for extraction pipeline and benchmarks."""

import json
from pathlib import Path

from core.mlflow_tracking import mlflow
from extraction.evaluation.benchmark.score import ExtractionBenchmarkSummary


def log_metric_mlflow(
    summary: ExtractionBenchmarkSummary,
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
