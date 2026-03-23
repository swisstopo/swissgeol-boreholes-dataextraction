"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from classification.classifiers.classifier import Classifier, ClassifierTypes
from classification.classifiers.classifier_factory import ClassifierFactory
from classification.evaluation.benchmark.score import (
    BenchmarkParams,
    ClassificationBenchmarkSummary,
    evaluate_all_predictions,
)
from classification.evaluation.benchmark.spec import BenchmarkSpec
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.data_loader import LayerInformation, prepare_classification_data
from classification.utils.data_utils import (
    get_data_class_count,
    get_data_language_count,
    write_predictions,
)
from core.benchmark_utils import (
    finalize_overall_summary,
    parent_input_key,
    run_multi_benchmark,
    short_metric_key,
)
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_parent_run, setup_mlflow_tracking
from core.pipeline_runner import PipelineRunResult, execute_pipeline_run

logger = logging.getLogger(__name__)


def setup_classification_mlflow_tracking(
    *,
    run_id: str | None,
    file_path: Path | None,
    ground_truth_path: Path | None,
    out_directory: Path | None,
    experiment_name: str = "Layer descriptions classification",
    runname: str | None = None,
    nested: bool = False,
) -> str:
    """Wraps setup_mlflow_tracking() with classification-specific tags and params."""
    return setup_mlflow_tracking(
        run_id=run_id,
        experiment_name=experiment_name,
        runname=runname,
        nested=nested,
        tags={
            "json_file_path": file_path,
            "ground_truth_path": ground_truth_path,
            "out_directory": out_directory,
        },
        params=None,
        include_git_info=False,
    )


def _setup_mlflow_parent_run(
    *,
    out_directory: Path,
    benchmarks: Sequence[BenchmarkSpec],
    experiment_name: str = "Layer descriptions classification",
    runid: str | None = None,
    runname: str | None = None,
) -> str:
    """Wraps setup_mlflow_parent_run() with classification specific input key and tags."""
    return setup_mlflow_parent_run(
        run_id=runid,
        experiment_name=experiment_name,
        runname=runname,
        parent_input_key=parent_input_key([Path(b.file_path) for b in benchmarks]),
        benchmarks=benchmarks,
        input_tag_name="json_file_path",
        ground_truth_path=None,
        out_directory=out_directory,
        include_git_info=False,
    )


def log_ml_flow_infos(
    file_path: Path,
    out_directory: Path,
    layer_descriptions: list[LayerInformation],
    classifier: Classifier,
    classification_system: str,
):
    """Logs information to mlflow, such as the number of sample, language distribution, classifier type and data.

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


def run_predictions(
    file_path: Path,
    ground_truth_path: Path | None,
    out_directory: Path,
    out_directory_bedrock: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
) -> tuple[list[LayerInformation], Classifier | None, int]:
    """Load data, run classification, and write predictions.

    This is the core prediction logic, decoupled from tracking and evaluation.

    Args:
        file_path (Path): Path to the JSON file containing material descriptions to classify.
        ground_truth_path (Path | None): Path to the ground truth file, or None for single-file mode.
        out_directory (Path): Path to output directory where predictions are written.
        out_directory_bedrock (Path): Path to output directory for Bedrock API files.
        file_subset_directory (Path | None): Path to the directory containing subset filenames.
        classifier_type (str): The classifier type to use (e.g. 'dummy', 'bert', 'baseline', 'bedrock').
        model_path (Path | None): Path to the trained model, if applicable.
        classification_system (str): The classification system to use (e.g. 'uscs', 'lithology').

    Returns:
        tuple[list[LayerInformation], Classifier | None, int]: The classified layer descriptions,
            the classifier instance used (or None if no data was found), and the number of
            unique documents processed.
    """
    classifier_type_instance = ClassifierTypes.infer_type(classifier_type.lower())
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(
        classification_system.lower()
    )

    logger.info(
        f"Loading data from {file_path}" + (f" and ground truth from {ground_truth_path}" if ground_truth_path else "")
    )
    layer_descriptions = prepare_classification_data(file_path, ground_truth_path, classification_system_cls)

    n_documents = len({layer.filename for layer in layer_descriptions})

    if not layer_descriptions:
        logger.warning("No data to classify.")
        return layer_descriptions, None, n_documents

    classifier = ClassifierFactory.create_classifier(
        classifier_type_instance, classification_system_cls, model_path, out_directory_bedrock
    )
    logger.info(
        f"Classifying layer description into {classification_system_cls.get_name()} classes "
        f"with {classifier.__class__.__name__}"
    )
    classifier.classify(layer_descriptions)
    write_predictions(layer_descriptions, out_directory)

    return layer_descriptions, classifier, n_documents


def start_pipeline(
    file_path: Path,
    ground_truth_path: Path | None,
    out_directory: Path,
    out_directory_bedrock: Path,
    predictions_path: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    resume: bool = False,
    runname: str | None = None,
    is_nested: bool = False,
) -> ClassificationBenchmarkSummary | None:
    """Main pipeline to classify the layer's soil descriptions."""
    out_directory.mkdir(parents=True, exist_ok=True)
    out_directory_bedrock.mkdir(parents=True, exist_ok=True)

    def _run_predictions(
        predictions_path_tmp: Path,
    ) -> PipelineRunResult[tuple[list[LayerInformation], Classifier | None]]:
        layer_descriptions, classifier, n_documents = run_predictions(
            file_path=file_path,
            ground_truth_path=ground_truth_path,
            out_directory=out_directory,
            out_directory_bedrock=out_directory_bedrock,
            classifier_type=classifier_type,
            model_path=model_path,
            classification_system=classification_system,
        )

        final_pred = out_directory / "class_predictions.json"
        copied_tmp: Path | None = None
        if final_pred.exists():
            shutil.copy(final_pred, predictions_path_tmp)
            copied_tmp = predictions_path_tmp

        return PipelineRunResult(
            predictions=(layer_descriptions, classifier),
            n_documents=n_documents,
            copy_source=copied_tmp,
        )

    def _evaluate(
        run_result: PipelineRunResult[tuple[list[LayerInformation], Classifier | None]],
    ) -> ClassificationBenchmarkSummary | None:
        layer_descriptions, _classifier = run_result.predictions

        if not layer_descriptions:
            logger.warning("No data to classify. Returning empty summary so parent can still aggregate n_documents.")
            return ClassificationBenchmarkSummary(
                file_path=str(file_path),
                ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
                n_documents=run_result.n_documents,
                classifier_type=classifier_type,
                model_path=str(model_path) if model_path else None,
                classification_system=classification_system,
                metrics={},
            )

        return evaluate_all_predictions(
            layer_descriptions=layer_descriptions,
            params=BenchmarkParams(
                file_path=file_path,
                ground_truth_path=ground_truth_path,
                classifier_type=classifier_type,
                model_path=model_path,
                classification_system=classification_system,
                n_documents=run_result.n_documents,
            ),
            out_directory=out_directory,
        )

    def _after_evaluation(
        run_result: PipelineRunResult[tuple[list[LayerInformation], Classifier | None]],
        summary: ClassificationBenchmarkSummary | None,
        _predictions_path_tmp: Path,
    ) -> None:
        layer_descriptions, classifier = run_result.predictions

        if mlflow and summary is not None and classifier is not None and layer_descriptions:
            log_ml_flow_infos(
                file_path=file_path,
                out_directory=out_directory,
                layer_descriptions=layer_descriptions,
                classifier=classifier,
                classification_system=classification_system,
            )

    return execute_pipeline_run(
        predictions_path=predictions_path,
        resume=resume,
        is_nested=is_nested,
        cleanup_mlflow_tmp=False,
        setup_mlflow_run=(
            lambda runid: setup_classification_mlflow_tracking(
                run_id=runid,
                file_path=file_path,
                ground_truth_path=ground_truth_path,
                out_directory=out_directory,
                experiment_name="Layer descriptions classification",
                runname=runname or file_path.name,
                nested=is_nested,
            )
            if mlflow
            else None
        ),
        run_predictions=_run_predictions,
        evaluate_predictions=_evaluate,
        after_evaluation=_after_evaluation,
        copy_predictions_to_final=True,
    )


def start_multi_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    out_directory_bedrock: Path,
    resume: bool = False,
) -> None:
    """Run multiple classification benchmarks in one execution."""
    multi_root = out_directory / "multi"
    multi_bedrock_root = out_directory_bedrock / "multi"
    parent_runid_tmp = multi_root / "mlflow_parent_runid.json.tmp"

    def setup_parent_run(parent_runid: str | None) -> str:
        return _setup_mlflow_parent_run(
            runid=parent_runid,
            out_directory=multi_root,
            benchmarks=benchmarks,
            experiment_name="Layer descriptions classification",
        )

    def run_single(spec: BenchmarkSpec) -> ClassificationBenchmarkSummary | None:
        logger.info("Running benchmark: %s", spec.name)

        bench_out = multi_root / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)

        bench_out_bedrock = multi_bedrock_root / spec.name
        bench_out_bedrock.mkdir(parents=True, exist_ok=True)
        bench_predictions_path = bench_out / "class_predictions.json"

        return start_pipeline(
            file_path=spec.file_path,
            ground_truth_path=spec.ground_truth_path,
            out_directory=bench_out,
            out_directory_bedrock=bench_out_bedrock,
            predictions_path=bench_predictions_path,
            classifier_type=classifier_type,
            model_path=model_path,
            classification_system=classification_system,
            resume=resume,
            runname=spec.name,
            is_nested=True,
        )

    def finalize(
        overall_results: list[tuple[str, ClassificationBenchmarkSummary | None]],
        root: Path,
    ) -> None:
        finalize_overall_summary(
            overall_results=overall_results,
            multi_root=root,
            aggregate_label="total/mean",
            metric_key_shortener=short_metric_key,
        )

    run_multi_benchmark(
        benchmarks=benchmarks,
        multi_root=multi_root,
        resume=resume,
        parent_runid_tmp=parent_runid_tmp,
        setup_parent_run=setup_parent_run if mlflow else None,
        run_single_benchmark=run_single,
        finalize_summary=finalize,
    )
