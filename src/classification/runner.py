"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from classification.classifiers.classifier import Classifier, ClassifierTypes
from classification.classifiers.classifier_factory import ClassifierFactory
from classification.evaluation.benchmark.score import ClassificationBenchmarkSummary, evaluate_all_predictions
from classification.evaluation.benchmark.spec import BenchmarkSpec
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.data_loader import LayerInformation, prepare_classification_data
from classification.utils.data_utils import (
    get_data_class_count,
    get_data_language_count,
    write_predictions,
)
from classification.utils.file_utils import read_params
from utils.benchmark_utils import _parent_input_directory_key, _short_metric_key
from utils.mlflow_tracking import mlflow

logger = logging.getLogger(__name__)

classification_params = read_params("classification_params.yml")


def _finalize_overall_summary(
    *,
    overall_results: list[tuple[str, None | ClassificationBenchmarkSummary]],
    multi_root: Path,
):
    """Write overall_summary.json and overall_summary.csv (+ mean row).

    Also logs overall aggregate metrics + artifacts to MLflow on the parent run (if enabled).
    """
    # --- JSON ---
    summary_path = multi_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        summary = [
            {"benchmark": benchmark, "summary": summary.model_dump() if summary else None}
            for benchmark, summary in overall_results
        ]
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- CSV ---
    summary_csv_path = multi_root / "overall_summary.csv"

    rows = []
    for benchmark, summary in overall_results:
        row: dict[str, any] = {"benchmark": benchmark}
        if summary is not None:
            row["ground_truth_path"] = summary.ground_truth_path
            row["n_documents"] = summary.n_documents
            row.update(summary.metrics_flat())
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by="benchmark")

    total_docs = df["n_documents"].sum()
    means = df.mean(numeric_only=True)
    agg_row = means.round(3)
    agg_row.at["benchmark"] = "total/mean"
    agg_row.at["n_documents"] = total_docs
    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    df.to_csv(summary_csv_path, index=False)

    # --- MLflow: overall mean metrics + artifacts on parent run ---
    if mlflow:
        for full_key, value in means.items():
            if pd.notna(value):
                short_key = _short_metric_key(full_key)
                mlflow.log_metric(short_key, value)

        if pd.notna(total_docs):
            mlflow.log_metric("total_n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")


def setup_mlflow_tracking(
    file_path: Path | None,
    out_directory: Path,
    file_subset_directory: Path | None,
    experiment_name: str = "Layer descriptions classification",
) -> None:
    """Set up MLFlow tracking.

    Args:
        file_path (Path): The path to the input file.
        out_directory (Path): The output directory.
        file_subset_directory (Path | None): The path to the subset directory.
        experiment_name (str): The MLflow experiment name.
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
    benchmarks: Sequence[BenchmarkSpec],
    experiment_name: str = "Layer descriptions classification",
) -> bool:
    """Start the parent MLflow run (multi-benchmark) and log global params once.

    Args:
        out_directory (Path): The output directory.
        benchmarks (Sequence[BenchmarkSpec]): The list of benchmark specifications.
        experiment_name: (str) The MLflow experiment name.

    Returns:
        bool: True if a parent run was started and must be closed by the caller.
    """
    if not mlflow:
        return False

    setup_mlflow_tracking(
        file_path=_parent_input_directory_key([Path(b.file_path) for b in benchmarks]),
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
    """Logs informations to mlflow, such as the number of sample, language distribution, classifier type and data.

    Args:
        file_path (Path): The path to the input file.
        out_directory (Path): The output directory where predictions are stored.
        layer_descriptions (list[LayerInformation]): The list of layer descriptions that were classified.
        classifier (Classifier): The classifier used for classification.
        classification_system (str): The classification system used for classification.
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
) -> ClassificationBenchmarkSummary | None:
    """Main pipeline to classify the layer's soil descriptions.

    Args:
        file_path (Path): Path to the json file we want to predict from.
        ground_truth_path (Path): Path to the ground truth file, if file_path is the predictions.
        out_directory (Path): Path to output directory.
        out_directory_bedrock (Path): Path to output directory for bedrock API files.
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        classifier_type (str): The classifier type to use.
        model_path (Path): Path to the trained model.
        classification_system (str): The classification system used to classify the data.
        mlflow_setup (bool): Whether to set up MLflow tracking in this function.
    """
    if ground_truth_path and file_subset_directory:
        logger.warning(
            "The provided subset directory will be ignored because descriptions are being loaded from the prediction "
            "file. All layers in the prediction file will be classified."
        )

    classifier_type_instance = ClassifierTypes.infer_type(classifier_type.lower())
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(
        classification_system.lower()
    )

    if mlflow and mlflow_setup:
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

    # write predictions
    write_predictions(layer_descriptions, out_directory)

    # centralize evaluation + per-class artifacts in score module
    summary = evaluate_all_predictions(
        layer_descriptions=layer_descriptions,
        file_path=file_path,
        ground_truth_path=ground_truth_path,
        out_directory=out_directory,
    )

    # If evaluate_all_predictions returned None, stop here
    if summary is None:
        if mlflow and mlflow_setup:
            mlflow.end_run()
        return None

    # Fill the fields that your current evaluate_all_predictions() left as placeholders
    summary = summary.model_copy(
        update={
            "file_subset_directory": str(file_subset_directory) if file_subset_directory else None,
            "classifier_type": classifier_type,
            "model_path": str(model_path) if model_path else None,
            "classification_system": classification_system,
        }
    )

    if mlflow:
        log_ml_flow_infos(file_path, out_directory, layer_descriptions, classifier, str(classification_system_cls))

    if mlflow and mlflow_setup:
        mlflow.end_run()

    return summary


def start_multi_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    out_directory_bedrock: Path,
):
    """Run multiple classification benchmarks in one execution.

    Args:
        benchmarks (Sequence[BenchmarkSpec]): A sequence of BenchmarkSpec, each specifying a benchmark to run.
        out_directory (Path): The root output directory where subdirectories for each benchmark will be created.
        classifier_type (str): The classifier type to use for all benchmarks.
        model_path (Path | None): Path to the trained model to use for all benchmarks.
        classification_system (str): The classification system to use for all benchmarks.
        out_directory_bedrock (Path): The root output directory for bedrock API files.
    """
    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)

    multi_bedrock_root = out_directory_bedrock / "multi"
    multi_bedrock_root.mkdir(parents=True, exist_ok=True)

    _setup_mlflow_parent_run(
        out_directory=out_directory,
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

            if mlflow:
                mlflow.start_run(run_name=spec.name, nested=True)
                setup_mlflow_tracking(
                    file_path=spec.file_subset_directory,
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

            # Per-benchmark summary artifact
            bench_summary_path = bench_out / "benchmark_summary.json"
            with open(bench_summary_path, "w", encoding="utf8") as f:
                json.dump(summary.model_dump() if summary else None, f, ensure_ascii=False, indent=2)

            if mlflow:
                mlflow.log_artifact(str(bench_summary_path), artifact_path="summary")

                if summary is not None:
                    if hasattr(summary, "metrics_flat"):
                        mlflow.log_metrics(summary.metrics_flat(short=True))

                    else:
                        # fallback: log raw metrics dict
                        mlflow.log_metrics({str(key): float(value) for key, value in (summary.metrics or {}).items()})

                mlflow.end_run()

        _finalize_overall_summary(overall_results=overall_results, multi_root=multi_root, mlflow_tracking=mlflow)
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
