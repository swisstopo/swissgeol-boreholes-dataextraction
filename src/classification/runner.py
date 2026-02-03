"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from classification.classifiers.classifier import Classifier, ClassifierTypes
from classification.classifiers.classifier_factory import ClassifierFactory
from classification.evaluation.benchmark.score import ClassificationBenchmarkSummary
from classification.evaluation.benchmark.spec import BenchmarkSpec
from classification.evaluation.evaluate import evaluate
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.data_loader import LayerInformation, prepare_classification_data
from classification.utils.data_utils import (
    get_data_class_count,
    get_data_language_count,
    write_per_language_per_class_predictions,
    write_predictions,
)
from swissgeol_doc_processing.utils.file_utils import read_params
from utils.benchmark_utils import _parent_input_directory_key

logger = logging.getLogger(__name__)

classification_params = read_params("classification_params.yml")

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow


def _finalize_overall_summary(
    *,
    overall_results: list[tuple[str, ClassificationBenchmarkSummary | None]],
    multi_root: Path,
    mlflow_tracking: bool,
    parent_active: bool,
) -> tuple[Path, Path]:
    """Write overall_summary.json and overall_summary.csv (+ mean row).

    Also logs artifacts + overall mean metrics to MLflow on the parent run (if enabled).
    """
    # --- JSON ---
    summary_json_path = multi_root / "overall_summary.json"
    payload = [
        {"benchmark": name, "summary": summary.model_dump() if summary else None} for name, summary in overall_results
    ]
    with open(summary_json_path, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- CSV ---
    rows: list[dict[str, Any]] = []
    for name, summary in overall_results:
        row: dict[str, Any] = {"benchmark": name}

        if summary is not None:
            row.update(
                {
                    "n_layers": summary.n_layers,
                    "file_path": summary.file_path,
                    "subset_dir": summary.file_subset_directory,
                    "ground_truth_path": summary.ground_truth_path,
                }
            )

            # use the same metric key-space as the child runs
            metrics_dict = (
                summary.metrics_flat(short=True) if hasattr(summary, "metrics_flat") else (summary.metrics or {})
            )

            for k, v in (metrics_dict or {}).items():
                row[f"metrics__{k}"] = v
        else:
            row.update(
                {
                    "n_layers": None,
                    "file_path": None,
                    "subset_dir": None,
                    "ground_truth_path": None,
                }
            )

        rows.append(row)

    summary_csv_path = multi_root / "overall_summary.csv"
    df = pd.DataFrame(rows).sort_values(by="benchmark")

    metric_cols = [c for c in df.columns if c.startswith("metrics__")]

    df_metrics = df.copy()
    df_metrics[metric_cols] = df_metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    df_metrics["n_layers"] = pd.to_numeric(df_metrics["n_layers"], errors="coerce")

    means: dict[str, float | None] = {}
    for c in metric_cols:
        col = df_metrics[c]
        means[c] = float(col.mean()) if col.notna().any() else None

    mean_row: dict[str, Any] = {
        "benchmark": "mean",
        "n_layers": int(df_metrics["n_layers"].sum()) if df_metrics["n_layers"].notna().any() else "",
        "file_path": "",
        "subset_dir": "",
        "ground_truth_path": "",
        **means,
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").round(4)
    df.to_csv(summary_csv_path, index=False)

    # --- MLflow parent logging ---
    if mlflow_tracking and parent_active:
        import mlflow

        overall_mean_metrics: dict[str, float] = {}
        for col_name, v in means.items():
            if v is None:
                continue

            # col_name: "metrics__<child_key>"
            child_key = col_name[len("metrics__") :]
            overall_mean_metrics[child_key] = float(v)

        if overall_mean_metrics:
            mlflow.log_metrics(overall_mean_metrics)

        total_layers = df_metrics["n_layers"].sum(skipna=True)
        if pd.notna(total_layers):
            mlflow.log_metric("overall/total_n_layers", float(total_layers))

        mlflow.log_artifact(str(summary_json_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")

    return summary_json_path, summary_csv_path


def setup_mlflow_tracking(
    file_path: Path | None,
    out_directory: Path,
    file_subset_directory: Path | None,
    experiment_name: str = "Layer descriptions classification",
):
    """Set up MLFlow tracking.

    Args:
        file_path: The path to the input file.
        out_directory: The output directory.
        file_subset_directory: The path to the subset directory.
        experiment_name: The MLflow experiment name.
    """
    mlflow.set_experiment(experiment_name)
    if mlflow.active_run() is None:
        mlflow.start_run()

    if file_path:
        mlflow.set_tag("json file_path", str(file_path))
    if file_subset_directory:
        mlflow.set_tag("file_subset_directory", str(file_subset_directory))
    mlflow.set_tag("out_directory", str(out_directory))


def _setup_mlflow_parent_run(
    *,
    out_directory: Path,
    mlflow_tracking: bool,
    benchmarks: Sequence[BenchmarkSpec],
    experiment_name: str = "Layer descriptions classification",
) -> bool:
    """Start the parent MLflow run (multi-benchmark) and log global params once.

    Args:
        out_directory: The output directory.
        mlflow_tracking: Whether MLflow tracking is enabled.
        benchmarks: The list of benchmark specifications.
        experiment_name: The MLflow experiment name.

    Returns: True if a parent run was started and must be closed by the caller.
    """
    if not mlflow_tracking:
        return False

    import mlflow

    setup_mlflow_tracking(
        file_path=_parent_input_directory_key(benchmarks),
        out_directory=out_directory,
        file_subset_directory=None,
        experiment_name=experiment_name,
    )
    mlflow.set_tag("run_type", "multi_benchmark")
    mlflow.set_tag("benchmarks", ",".join([b.name for b in benchmarks]))

    return True


def log_ml_flow_infos(
    file_path: Path,
    out_directory: Path,
    layer_descriptions: list[LayerInformation],
    classifier: Classifier,
    classification_system: str,
):
    """Logs informations to mlflow, such as the number of sample, language distribution, classifier type and data."""
    # Log dataset statistics
    mlflow.log_param("dataset_size", len(layer_descriptions))

    # Log language distribution
    for language, count in get_data_language_count(layer_descriptions).items():
        mlflow.log_param(f"language_{language}_count", count)

    # Log class distribution
    for class_, count in get_data_class_count(layer_descriptions).items():
        mlflow.log_param(f"class_{class_}_count", count)

    # Log the classification systems used
    mlflow.log_param("classification_systems", classification_system)

    # Log classifier name and parameters
    mlflow.log_param("classifier_type", classifier.__class__.__name__)
    classifier.log_params()

    # Log input data and output predictions
    mlflow.log_artifact(str(file_path), "input_data")
    mlflow.log_artifact(f"{out_directory}/class_predictions.json", "predictions_json")

    # log output prediction artifacts detailed for each class
    pred_dir = os.path.join(out_directory, "predictions_per_class")
    for language in ["global", *classification_params["supported_language"]]:
        overview_path = os.path.join(pred_dir, language, "overview.csv")
        mlflow.log_artifact(overview_path, f"predictions_per_class_json/{language}")
        for first_key in ["ground_truth", "prediction"]:
            language_first_key_dir = os.path.join(pred_dir, language, f"group_by_{first_key}")
            artifact_directory = f"predictions_per_class_json/{language}/group_by_{first_key}"
            for file in os.listdir(language_first_key_dir):
                file_path = os.path.join(language_first_key_dir, file)
                mlflow.log_artifact(file_path, artifact_directory)


def start_pipeline(
    file_path: Path,
    ground_truth_path: Path,
    out_directory: Path,
    out_directory_bedrock: Path,
    file_subset_directory: Path,
    classifier_type: str,
    model_path: Path,
    classification_system: str,
    mlflow_setup: bool = True,
) -> ClassificationBenchmarkSummary | None:
    """Main pipeline to classify the layer's soil descriptions."""
    if ground_truth_path and file_subset_directory:
        logger.warning(
            "The provided subset directory will be ignored because descriptions are being loaded from the prediction "
            "file. All layers in the prediction file will be classified."
        )

    classifier_type_instance = ClassifierTypes.infer_type(classifier_type.lower())
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(
        classification_system.lower()
    )

    if mlflow_tracking and mlflow_setup:
        setup_mlflow_tracking(file_path, out_directory, file_subset_directory)

    logger.info(
        f"Loading data from {file_path}" + (f" and ground truth from {ground_truth_path}" if ground_truth_path else "")
    )
    layer_descriptions = prepare_classification_data(
        file_path, ground_truth_path, file_subset_directory, classification_system_cls
    )
    if not layer_descriptions:
        logger.warning("No data to classify.")
        return None

    classifier = ClassifierFactory.create_classifier(
        classifier_type_instance, classification_system_cls, model_path, out_directory_bedrock
    )

    logger.info(
        f"Classifying layer description into {classification_system_cls.get_name()} classes "
        f"with {classifier.__class__.__name__}"
    )
    classifier.classify(layer_descriptions)

    logger.info("Evaluating predictions")
    classification_metrics = evaluate(layer_descriptions)
    logger.info("classification metrics: %s", classification_metrics.to_json())
    logger.debug("classification metrics per class: %s", classification_metrics.to_json_per_class())

    write_predictions(layer_descriptions, out_directory)
    write_per_language_per_class_predictions(layer_descriptions, classification_metrics, out_directory)

    if mlflow_tracking:
        log_ml_flow_infos(file_path, out_directory, layer_descriptions, classifier, classification_system_cls)

    if mlflow_tracking and mlflow_setup:
        mlflow.end_run()

    return ClassificationBenchmarkSummary(
        file_path=str(file_path),
        ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
        file_subset_directory=str(file_subset_directory) if file_subset_directory else None,
        n_layers=len(layer_descriptions),
        classifier_type=classifier_type,
        model_path=str(model_path) if model_path else None,
        classification_system=classification_system,
        metrics={
            **(classification_metrics).to_json(),
            **(classification_metrics).to_json_per_class(),
        },
    )


def start_multi_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    out_directory_bedrock: Path,
    mlflow_tracking: bool = False,
    classification_params: dict | None = None,
):
    """Run multiple classification benchmarks in one execution."""
    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)

    multi_bedrock_root = out_directory_bedrock / "multi"
    multi_bedrock_root.mkdir(parents=True, exist_ok=True)

    classification_params = classification_params or {}

    parent_active = _setup_mlflow_parent_run(
        out_directory=out_directory,
        mlflow_tracking=mlflow_tracking,
        benchmarks=benchmarks,
        experiment_name="Layer descriptions classification",
    )

    overall_results: list[tuple[str, ClassificationBenchmarkSummary | None]] = []

    try:
        for spec in benchmarks:
            logger.info("Running benchmark: %s", spec.name)

            bench_out = multi_root / spec.name
            bench_out.mkdir(parents=True, exist_ok=True)

            bench_out_bedrock = multi_bedrock_root / spec.name
            bench_out_bedrock.mkdir(parents=True, exist_ok=True)

            if mlflow_tracking:
                import mlflow

                mlflow.start_run(run_name=spec.name, nested=True)
                setup_mlflow_tracking(
                    file_path=spec.file_path,
                    out_directory=out_directory,
                    file_subset_directory=None,
                )

            summary = start_pipeline(
                file_path=spec.file_path,
                ground_truth_path=spec.ground_truth_path,
                out_directory=bench_out,
                out_directory_bedrock=bench_out_bedrock,
                file_subset_directory=spec.file_subset_directory,
                classifier_type=classifier_type,
                model_path=model_path,
                classification_system=classification_system,
                mlflow_setup=False,  # multi-benchmark owns the run lifecycle
            )

            overall_results.append((spec.name, summary))

            # Per-benchmark summary artifact (always written)
            bench_summary_path = bench_out / "benchmark_summary.json"
            with open(bench_summary_path, "w", encoding="utf8") as f:
                json.dump(summary.model_dump() if summary else None, f, ensure_ascii=False, indent=2)

            if mlflow_tracking:
                import mlflow

                mlflow.log_artifact(str(bench_summary_path), artifact_path="summary")

                if summary is not None:
                    if hasattr(summary, "metrics_flat"):
                        mlflow.log_metrics(summary.metrics_flat(short=True))

                    else:
                        # fallback: log raw metrics dict
                        mlflow.log_metrics({str(k): float(v) for k, v in (summary.metrics or {}).items()})

                mlflow.end_run()

        _finalize_overall_summary(
            overall_results=overall_results,
            multi_root=multi_root,
            mlflow_tracking=mlflow_tracking,
            parent_active=parent_active,
        )
    finally:
        if mlflow_tracking and parent_active:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.end_run()
