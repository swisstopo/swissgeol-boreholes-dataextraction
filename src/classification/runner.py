"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
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
    short_metric_key,
)
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_parent_run, setup_mlflow_tracking
from core.pipeline_runner import MultiBenchmarkRunner, PipelineRunner, PipelineRunResult

logger = logging.getLogger(__name__)

_ClassificationResult = tuple[list[LayerInformation], Classifier | None]


@dataclass
class ClassificationOptions:
    """Options shared between single and multi-benchmark classification runners."""

    classifier_type: str
    model_path: Path | None
    classification_system: str


def run_classification_predictions(
    file_path: Path,
    ground_truth_path: Path | None,
    out_directory: Path,
    out_directory_bedrock: Path,
    options: ClassificationOptions,
) -> tuple[list[LayerInformation], Classifier | None, int]:
    """Load data, run classification, and write predictions.

    This is the core prediction logic, decoupled from tracking and evaluation.

    Args:
        file_path (Path): Path to the JSON file containing material descriptions to classify.
        ground_truth_path (Path | None): Path to the ground truth file, or None for single-file mode.
        out_directory (Path): Path to output directory where predictions are written.
        out_directory_bedrock (Path): Path to output directory for Bedrock API files.
        options (ClassificationOptions): Classification run options.

    Returns:
        tuple[list[LayerInformation], Classifier | None, int]: The classified layer descriptions,
            the classifier instance used (or None if no data was found), and the number of
            unique documents processed.
    """
    classifier_type_instance = ClassifierTypes.infer_type(options.classifier_type.lower())
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(
        options.classification_system.lower()
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
        classifier_type_instance, classification_system_cls, options.model_path, out_directory_bedrock
    )
    logger.info(
        f"Classifying layer description into {classification_system_cls.get_name()} classes "
        f"with {classifier.__class__.__name__}"
    )
    classifier.classify(layer_descriptions)
    write_predictions(layer_descriptions, out_directory)

    return layer_descriptions, classifier, n_documents


@dataclass(kw_only=True)
class ClassificationPipelineRunner(PipelineRunner[_ClassificationResult, ClassificationBenchmarkSummary]):
    """Runs the layer descriptions classification pipeline."""

    file_path: Path
    ground_truth_path: Path | None
    out_directory: Path
    out_directory_bedrock: Path
    options: ClassificationOptions
    runname: str | None = None
    cleanup_mlflow_tmp: bool = False
    copy_predictions_to_final: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.out_directory.mkdir(parents=True, exist_ok=True)
        self.out_directory_bedrock.mkdir(parents=True, exist_ok=True)

    def setup_mlflow_run(self, runid: str | None) -> str:
        return setup_mlflow_tracking(
            run_id=runid,
            experiment_name="Layer descriptions classification",
            runname=self.runname,
            nested=self.is_nested,
            tags={
                "json_file_path": self.file_path,
                "ground_truth_path": self.ground_truth_path,
                "out_directory": self.out_directory,
            },
            params=None,
            include_git_info=False,
        )

    def run_predictions(self, predictions_path_tmp: Path) -> PipelineRunResult[_ClassificationResult]:
        layer_descriptions, classifier, n_documents = run_classification_predictions(
            file_path=self.file_path,
            ground_truth_path=self.ground_truth_path,
            out_directory=self.out_directory,
            out_directory_bedrock=self.out_directory_bedrock,
            options=self.options,
        )

        final_pred = self.out_directory / "class_predictions.json"
        if final_pred.exists():
            shutil.copy(final_pred, predictions_path_tmp)

        return PipelineRunResult(
            result=(layer_descriptions, classifier),
            n_documents=n_documents,
        )

    def evaluate(self, run_result: PipelineRunResult[_ClassificationResult]) -> ClassificationBenchmarkSummary | None:
        layer_descriptions, _classifier = run_result.result

        if not layer_descriptions:
            logger.warning("No data to classify. Returning empty summary so parent can still aggregate n_documents.")
            return ClassificationBenchmarkSummary(
                file_path=str(self.file_path),
                ground_truth_path=str(self.ground_truth_path) if self.ground_truth_path else None,
                n_documents=run_result.n_documents,
                classifier_type=self.options.classifier_type,
                model_path=str(self.options.model_path) if self.options.model_path else None,
                classification_system=self.options.classification_system,
                metrics={},
            )

        return evaluate_all_predictions(
            layer_descriptions=layer_descriptions,
            params=BenchmarkParams(
                file_path=self.file_path,
                ground_truth_path=self.ground_truth_path,
                classifier_type=self.options.classifier_type,
                model_path=self.options.model_path,
                classification_system=self.options.classification_system,
                n_documents=run_result.n_documents,
            ),
            out_directory=self.out_directory,
        )

    def after_evaluation(
        self,
        run_result: PipelineRunResult[_ClassificationResult],
        summary: ClassificationBenchmarkSummary | None,
        _predictions_path_tmp: Path,
    ) -> None:
        layer_descriptions, classifier = run_result.result

        if mlflow and summary is not None and classifier is not None and layer_descriptions:
            mlflow.log_param("dataset_size", len(layer_descriptions))
            for language, count in get_data_language_count(layer_descriptions).items():
                mlflow.log_param(f"language_{language}_count", count)
            for class_, count in get_data_class_count(layer_descriptions).items():
                mlflow.log_param(f"class_{class_}_count", count)
            mlflow.log_param("classification_systems", self.options.classification_system)
            mlflow.log_param("classifier_type", classifier.__class__.__name__)
            classifier.log_params()
            mlflow.log_artifact(str(self.file_path), "input_data")
            mlflow.log_artifact(f"{self.out_directory}/class_predictions.json", "predictions_json")


@dataclass(kw_only=True)
class ClassificationBenchmarkRunner(MultiBenchmarkRunner[BenchmarkSpec, ClassificationBenchmarkSummary]):
    """Orchestrates multiple classification benchmarks with shared MLflow parent tracking."""

    options: ClassificationOptions
    out_directory_bedrock: Path

    def setup_parent_run(self, runid: str | None) -> str:
        return setup_mlflow_parent_run(
            run_id=runid,
            experiment_name="Layer descriptions classification",
            parent_input_key=parent_input_key([Path(b.file_path) for b in self.benchmarks]),
            benchmarks=self.benchmarks,
            input_tag_name="json_file_path",
            ground_truth_path=None,
            out_directory=self.multi_root.parent,
            include_git_info=False,
        )

    def run_single(self, spec: BenchmarkSpec) -> ClassificationBenchmarkSummary | None:
        logger.info("Running benchmark: %s", spec.name)

        bench_out = self.multi_root / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)

        bench_out_bedrock = self.out_directory_bedrock / spec.name
        bench_out_bedrock.mkdir(parents=True, exist_ok=True)

        return ClassificationPipelineRunner(
            predictions_path=bench_out / "class_predictions.json",
            resume=self.resume,
            is_nested=True,
            file_path=spec.file_path,
            ground_truth_path=spec.ground_truth_path,
            out_directory=bench_out,
            out_directory_bedrock=bench_out_bedrock,
            options=self.options,
            runname=spec.name,
        ).execute()

    def finalize_summary(
        self,
        overall_results: list[tuple[str, ClassificationBenchmarkSummary | None]],
        root: Path,
    ) -> None:
        finalize_overall_summary(
            overall_results=overall_results,
            multi_root=root,
            aggregate_label="total/mean",
            metric_key_shortener=short_metric_key,
        )
