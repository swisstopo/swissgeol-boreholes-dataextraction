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
from swissgeol_doc_processing.utils.file_utils import flatten, read_params

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

classification_params = read_params("classification_params.yml")

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow


def _flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics dict into {"metrics/global_macro_f1": 0.63, ...}.

    Keeps only numeric values (int/float or strings convertible to float).

    Args:
        d: Nested metrics dictionary.
        prefix: Prefix to add to keys (used for recursion).

    Returns:
        Flattened dictionary with numeric values only.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"

        if v is None:
            continue
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        elif isinstance(v, (int | float)):
            out[key] = float(v)
        elif isinstance(v, str):
            try:
                out[key] = float(v)
            except ValueError:
                continue

    return out


def _collect_metric_keys(overall_results: list[dict[str, Any]]) -> list[str]:
    """Determine the union of 'metrics' keys across all benchmarks.

    This way, the CSV has stable columns even when some benchmarks emit extra keys.

    Args:
        overall_results: List of benchmark results.

    Returns:
        Sorted list of unique metric keys found.
    """
    keys: set[str] = set()

    for result in overall_results:
        summary = result.get("summary")
        if not isinstance(summary, dict):
            continue

        metrics = summary.get("metrics")
        if isinstance(metrics, dict):
            keys.update(metrics.keys())

    return sorted(keys)


def _make_overall_summary_rows(overall_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create rows for the overall summary CSV from benchmark results.

    Args:
        overall_results: List of benchmark results.

    Returns:
        List of rows for the overall summary CSV.
    """
    metric_keys = _collect_metric_keys(overall_results)
    rows: list[dict[str, Any]] = []

    for result in overall_results:
        summary = result.get("summary") or {}
        metrics = summary.get("metrics") if isinstance(summary, dict) else {}

        base: dict[str, Any] = {
            "benchmark": result.get("benchmark"),
            "n_layers": summary.get("n_layers") if isinstance(summary, dict) else None,
            "file_path": summary.get("file_path") if isinstance(summary, dict) else None,
            "subset_dir": summary.get("file_subset_directory") if isinstance(summary, dict) else None,
            "ground_truth_path": summary.get("ground_truth_path") if isinstance(summary, dict) else None,
        }

        # Stable metric columns (even if missing in some benchmarks)
        if isinstance(metrics, dict):
            for key in metric_keys:
                base[f"metrics__{key}"] = metrics.get(key)
        else:
            for key in metric_keys:
                base[f"metrics__{key}"] = None
        rows.append(base)

    return rows


def _setup_mlflow_parent_run(
    *,
    mlflow_tracking: bool,
    benchmarks: Sequence[BenchmarkSpec],
    classifier_type: str,
    classification_system: str,
    model_path: Path | None,
    classification_params: dict[str, Any],
) -> bool:
    """Start the parent MLflow run (multi-benchmark) and log global params once.

    Returns True if a parent run was started and must be closed by the caller.

    Args:
        mlflow_tracking: Whether to log to MLflow.
        benchmarks: List of benchmark specs.
        classifier_type: Type of classifier used.
        classification_system: System used for classification.
        model_path: Path to the model file (if applicable).
        classification_params: Parameters used for classification.

    Returns:
        True if a parent MLflow run was started, False otherwise.
    """
    if not mlflow_tracking:
        return False

    import mlflow

    mlflow.set_experiment("Layer descriptions classification")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.start_run(run_name="multi-benchmark")
    mlflow.set_tag("run_type", "multi_benchmark")
    mlflow.set_tag("benchmarks", ",".join([b.name for b in benchmarks]))
    mlflow.set_tag("classifier_type", classifier_type)
    mlflow.set_tag("classification_system", classification_system)
    if model_path:
        mlflow.set_tag("model_path", str(model_path))

    if classification_params:
        mlflow.log_params(flatten(classification_params))

    return True


def _finalize_overall_summary(
    *,
    overall_results: list[dict[str, Any]],
    multi_root: Path,
    mlflow_tracking: bool,
    parent_active: bool,
) -> tuple[Path, Path]:
    """Write overall_summary.json and overall_summary.csv (+ mean row).

    Also logs artifacts + overall mean metrics to MLflow on the parent run (if enabled).

    Args:
        overall_results: List of benchmark results.
        multi_root: Root directory for multi-benchmark outputs.
        mlflow_tracking: Whether MLflow tracking is enabled.
        parent_active: Whether the parent MLflow run is active.

    Returns:
        Paths to the written overall_summary.json and overall_summary.csv files.
    """
    #  JSON
    summary_json_path = multi_root / "overall_summary.json"
    with open(summary_json_path, "w", encoding="utf8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    #  CSV
    summary_csv_path = multi_root / "overall_summary.csv"
    rows = _make_overall_summary_rows(overall_results)
    df = pd.DataFrame(rows).sort_values(by="benchmark")

    metric_cols = [c for c in df.columns if c.startswith("metrics__")]
    df_metrics = df.copy()

    df_metrics[metric_cols] = df_metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    df_metrics["n_layers"] = pd.to_numeric(df_metrics["n_layers"], errors="coerce")

    # Per-metric means (across benchmarks)
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

    # Round numeric metric cols
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").round(4)

    df.to_csv(summary_csv_path, index=False)

    #  MLflow parent logging
    if mlflow_tracking and parent_active:
        import mlflow

        overall_mean_metrics: dict[str, float] = {}
        for k, v in means.items():
            if v is None:
                continue
            overall_key = "overall_mean/" + k.replace("__", "/")
            overall_mean_metrics[overall_key] = float(v)

        if overall_mean_metrics:
            mlflow.log_metrics(overall_mean_metrics)

        total_layers = df_metrics["n_layers"].sum(skipna=True)
        if pd.notna(total_layers):
            mlflow.log_metric("overall/total_n_layers", float(total_layers))

        mlflow.log_artifact(str(summary_json_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")

    return summary_json_path, summary_csv_path


def setup_mlflow_tracking(
    file_path: Path,
    out_directory: Path,
    file_subset_directory: Path,
    experiment_name: str = "Layer descriptions classification",
):
    """Set up MLFlow tracking.

    Args:
        file_path: Path to the input data file.
        out_directory: Path to the output directory.
        file_subset_directory: Path to the subset directory.
        experiment_name: Name of the MLFlow experiment.
    """
    if mlflow.active_run():
        mlflow.end_run()  # Ensure the previous run is closed
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("json file_path", str(file_path))
    mlflow.set_tag("out_directory", str(out_directory))
    mlflow.set_tag("file_subset_directory", str(file_subset_directory))


def log_ml_flow_infos(
    file_path: Path,
    out_directory: Path,
    layer_descriptions: list[LayerInformation],
    classifier: Classifier,
    classification_system: str,
):
    """Logs informations to mlflow, such as the number of sample, laguage distribution, classifier type and data.

    Args:
        file_path: Path to the input data file.
        out_directory: Path to the output directory.
        layer_descriptions: List of LayerInformation objects containing the data.
        classifier: The classifier used for classification.
        classification_system: The classification system used.
    """
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
):
    """Main pipeline to classify the layer's soil descriptions.

    Args:
        file_path (Path): Path to the json file we want to predict from.
        ground_truth_path (Path): Path the the ground truth file, if file_path is the predictions.
        out_directory (Path): Path to output directory
        out_directory_bedrock (Path): Path to output directory for bedrock API files
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        classifier_type (str): The classifier type to use.
        model_path (Path): Path to the trained model.
        classification_system (str): The classification system used to classify the data.
        mlflow_setup (bool): Whether to setup mlflow tracking.
    """
    if ground_truth_path and file_subset_directory:
        logger.warning(
            "The provided subset directory will be ignored because description are being loaded from the prediction"
            " file. All layers in the prediction file will be classified."
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
        return

    classifier = ClassifierFactory.create_classifier(
        classifier_type_instance, classification_system_cls, model_path, out_directory_bedrock
    )

    # classify
    logger.info(
        f"Classifying layer description into {classification_system_cls.get_name()} classes "
        f"with {classifier.__class__.__name__}"
    )
    classifier.classify(layer_descriptions)

    logger.info("Evaluating predictions")
    classification_metrics = evaluate(layer_descriptions)
    logger.info(f"classification metrics: {classification_metrics.to_json()}")
    logger.debug(f"classification metrics per class: {classification_metrics.to_json_per_class()}")

    write_predictions(layer_descriptions, out_directory)
    write_per_language_per_class_predictions(layer_descriptions, classification_metrics, out_directory)

    if mlflow_tracking:
        log_ml_flow_infos(file_path, out_directory, layer_descriptions, classifier, classification_system_cls)

    if mlflow_tracking and mlflow_setup:
        mlflow.end_run()

    summary = {
        "file_path": str(file_path),
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "file_subset_directory": str(file_subset_directory) if file_subset_directory else None,
        "n_layers": len(layer_descriptions),
        "classifier_type": classifier_type,
        "model_path": str(model_path) if model_path else None,
        "classification_system": classification_system,
        "metrics": classification_metrics.to_json(),
    }

    return summary


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
    """Run multiple classification benchmarks in one execution.

    Args:
        benchmarks: List of BenchmarkSpec objects defining the benchmarks to run.
        out_directory: Root output directory for multi-benchmark results.
        classifier_type: Type of classifier to use.
        model_path: Path to the model file (if applicable).
        classification_system: Classification system to use.
        out_directory_bedrock: Root output directory for bedrock API files.
        mlflow_tracking: Whether to enable MLflow tracking.
        classification_params: Additional classification parameters.
    """
    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)

    multi_bedrock_root = out_directory_bedrock / "multi"
    multi_bedrock_root.mkdir(parents=True, exist_ok=True)

    classification_params = classification_params or {}

    parent_active = _setup_mlflow_parent_run(
        mlflow_tracking=mlflow_tracking,
        benchmarks=benchmarks,
        classifier_type=classifier_type,
        classification_system=classification_system,
        model_path=model_path,
        classification_params=classification_params,
    )

    overall_results: list[dict[str, Any]] = []

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
                mlflow.set_tag("benchmark_name", spec.name)
                mlflow.set_tag("file_path", str(spec.file_path))
                mlflow.set_tag(
                    "file_subset_directory",
                    str(spec.file_subset_directory) if spec.file_subset_directory else "",
                )
                mlflow.set_tag(
                    "ground_truth_path",
                    str(spec.ground_truth_path) if spec.ground_truth_path else "",
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

            overall_results.append({"benchmark": spec.name, "summary": summary})

            # Per-benchmark summary artifact (always written)
            bench_summary_path = bench_out / "benchmark_summary.json"
            with open(bench_summary_path, "w", encoding="utf8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            if mlflow_tracking:
                import mlflow

                mlflow.log_artifact(str(bench_summary_path), artifact_path="summary")

                if isinstance(summary, dict):
                    flat = _flatten_metrics(summary, prefix="")
                    if flat:
                        mlflow.log_metrics(flat)

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
