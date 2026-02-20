"""Evaluate the predictions against the ground truth."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from classification.evaluation.evaluate import evaluate
from classification.utils.data_utils import write_per_language_per_class_predictions
from core.mlflow_tracking import mlflow

logger = logging.getLogger(__name__)


class ClassificationBenchmarkSummary(BaseModel):
    """Helper class containing a summary of all the results of a single classification benchmark."""

    file_path: str
    ground_truth_path: str | None
    file_subset_directory: str | None
    n_documents: int
    classifier_type: str
    model_path: str | None
    classification_system: str
    metrics: dict[str, float]

    def metrics_flat(self, prefix: str = "metrics", short: bool = False) -> dict[str, float] | None:
        """Flatten the metrics dictionary to a single level.

        Args:
            prefix (str): The prefix to use for the flattened keys.
            short (bool): Whether to use short keys (i.e. without the prefix).

        Returns:
            dict[str, float]: The flattened metrics dictionary.
        """
        if short:
            return dict(self.metrics)

        return {f"{prefix}/{k}": v for k, v in self.metrics.items()}


@dataclass(frozen=True)
class BenchmarkParams:
    """This class summarizes benchmark parameters which are needed for ClassificationBenchmarkSummary."""

    file_path: Path
    ground_truth_path: Path | None
    file_subset_directory: Path | None
    classifier_type: str
    model_path: Path | None
    classification_system: str
    n_documents: int


def evaluate_all_predictions(
    *,
    layer_descriptions,
    params: BenchmarkParams,
    out_directory: Path,
) -> ClassificationBenchmarkSummary | None:
    """Classification equivalent of extraction.score.evaluate_all_predictions().

    - Computes metrics (via `evaluate`)
    - Writes evaluation artifacts

    Args:
        layer_descriptions (list[LayerInformation]): The list of layer descriptions that were classified.
        params (BenchmarkParams): The benchmark parameters.
        out_directory (Path): The output directory where evaluation artifacts are written.

    Returns:
        ClassificationBenchmarkSummary | None: A JSON-serializable ClassificationBenchmarkSummary
        that can be used by multi-benchmark runners.
    """
    if not layer_descriptions:
        logger.warning("No data to evaluate.")
        return None

    logger.info("Evaluating predictions")
    classification_metrics = evaluate(layer_descriptions)

    # --- Write evaluation artifacts ---
    # 1) A single JSON dump of metrics
    metrics_path = out_directory / "classification_metrics.json"
    with open(metrics_path, "w", encoding="utf8") as f:
        json.dump(
            {
                "global": classification_metrics.to_json(),
                "per_class": classification_metrics.to_json_per_class(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2) Existing “per language / per class” outputs
    write_per_language_per_class_predictions(layer_descriptions, classification_metrics, out_directory)

    # --- MLflow logging ---
    if mlflow:
        mlflow.log_artifact(str(metrics_path), artifact_path="summary")
        pred_dir = out_directory / "predictions_per_class"
        if pred_dir.exists():
            mlflow.log_artifacts(str(pred_dir), artifact_path="predictions_per_class")

    # --- Return summary object ---
    return ClassificationBenchmarkSummary(
        file_path=str(params.file_path),
        ground_truth_path=str(params.ground_truth_path) if params.ground_truth_path else None,
        file_subset_directory=str(params.file_subset_directory) if params.file_subset_directory else None,
        n_documents=params.n_documents,
        classifier_type=params.classifier_type,
        model_path=str(params.model_path) if params.model_path else None,
        classification_system=params.classification_system,
        metrics={
            **classification_metrics.to_json(),
            **classification_metrics.to_json_per_class(),
        },
    )
