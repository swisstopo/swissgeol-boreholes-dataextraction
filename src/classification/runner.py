"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from classification.classifiers.classifier import Classifier, ClassifierTypes
from classification.classifiers.classifier_factory import ClassifierFactory
from classification.evaluation.benchmark.score import (
    BenchmarkParams,
    ClassificationBenchmarkSummary,
    evaluate_all_predictions,
)
from classification.evaluation.benchmark.spec import BenchmarkSpec
from classification.utils.data_utils import (
    get_data_class_count,
    get_data_language_count,
    write_predictions,
)
from classification.utils.datasets import ExistingClassificationSystems
from classification.utils.datasets.classification import (
    ClassificationSystem,
    GroundTruthBoreholeWithLanguage,
    LayerInformation,
    deterministic_hash_ratio,
    split_samples,
)
from core.ground_truth import GroundTruth, GroundTruthBorehole
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_tracking
from core.pipeline_runner import MultiBenchmarkRunner, PipelineRunner, PipelineRunResult
from extraction.features.classification_text import extract_borehole_texts
from extraction.runner import read_json_predictions

logger = logging.getLogger(__name__)

_ClassificationResult = tuple[list[LayerInformation], Classifier | None]


@dataclass
class ClassificationOptions:
    """Options shared between single and multi-benchmark classification runners."""

    classifier_type: str
    model_path: str | None
    classification_system: str
    backbone_path: Path | None = None
    tokenizer_path: Path | None = None
    predict_all: bool = False
    document_texts_paths: tuple[Path, ...] = ()


def _expand_json_paths(paths: tuple[Path, ...]) -> list[Path]:
    """Expand each given path into its JSON file(s): itself if a file, or its *.json children if a directory."""
    return [json_path for path in paths for json_path in ([path] if path.is_file() else sorted(path.glob("*.json")))]


def _is_pdf_path(path: Path) -> bool:
    """Whether `path` (a file or directory) refers to PDF(s) rather than a ground truth/predictions JSON."""
    if path.is_file():
        return path.suffix.lower() == ".pdf"
    return any(path.glob("*.pdf"))


def _load_ground_truth_dict(paths: tuple[Path, ...]) -> dict[str, list[GroundTruthBorehole]]:
    """Load and merge ground truth boreholes from one or more JSON files/directories."""
    merged: dict[str, list[GroundTruthBorehole]] = {}
    for json_path in _expand_json_paths(paths):
        merged.update(GroundTruth(json_path).ground_truth)
    return merged


def _extract_file_text(pdf_path: Path) -> str:
    """Extract a single file-level text for `pdf_path`, merging all its boreholes (dropping exact repeats)."""
    borehole_texts = extract_borehole_texts(pdf_path, pdf_path.name)
    return "\n".join(dict.fromkeys(borehole_text.text for borehole_text in borehole_texts))


def _test_split_filenames(filenames: Iterable[str]) -> set[str]:
    """Filenames landing in the held-out test split, by the same deterministic filename hash as split_samples.

    Filename hashing needs no document text, so this lets callers restrict expensive text extraction to
    only the ~15% of files that will actually survive `split_samples()`, instead of extracting everything
    and discarding most of it. 0.15 mirrors `split_samples`'s default `rtest` -- there's no CLI knob to
    configure a different ratio, so this can't drift out of sync with a value nobody can currently set.
    """
    return {filename for filename in filenames if deterministic_hash_ratio(filename) < 0.15}


def _load_document_texts(paths: tuple[Path, ...], restrict_to: set[str] | None = None) -> dict[str, str]:
    """Load and merge a filename -> text mapping from one or more sources.

    Each path is either a precomputed JSON file/directory of JSON files (fast, e.g. training's cached
    *_filtered_text.json), or a PDF file/directory of PDFs, extracted live via the same extraction
    pipeline used for raw-PDF inference and the classify_borehole_type API endpoint. Mixing both kinds
    across multiple -dt values is fine, e.g. cached JSON for large datasets, live PDFs for the rest.

    Args:
        paths: The -dt sources to load/merge.
        restrict_to: If given, PDF sources only extract files whose name is in this set (see
            `_test_split_filenames`) -- skipped entirely otherwise, since extraction is the expensive
            part. JSON sources are loaded in full regardless: parsing the file already reads it whole,
            so pre-filtering wouldn't save any work.
    """
    merged: dict[str, str] = {}
    for path in paths:
        if _is_pdf_path(path):
            pdf_paths = [path] if path.is_file() else sorted(path.glob("*.pdf"))
            if restrict_to is not None:
                pdf_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.name in restrict_to]
            for pdf_path in tqdm(pdf_paths, desc=f"Extracting text ({path.name})", unit="file"):
                merged[pdf_path.name] = _extract_file_text(pdf_path)
        else:
            for json_path in _expand_json_paths((path,)):
                with open(json_path, encoding="utf-8") as f:
                    merged.update(json.load(f))
    return merged


def _load_layer_descriptions(
    file_paths: tuple[Path, ...],
    classification_system_cls: type[ClassificationSystem],
    options: ClassificationOptions,
    document_text: dict[str, str] | None = None,
) -> tuple[list[LayerInformation], bool]:
    """Load layer descriptions either as predictions or as ground truth (test set).

    Args:
        file_paths: One or more ground truth/predictions JSON files (or directories of such files),
            merged together as a single dataset.
        classification_system_cls: The classification system to load samples for.
        options: Classification run options.
        document_text: For document-level systems, a filename -> text mapping (see
            `ClassificationSystem.process`). Ignored for layer-level systems.

    Returns:
        tuple[list[LayerInformation], bool]: The layer descriptions to classify, and whether
            they came from prediction data (True) or ground truth data (False).
    """
    try:
        logger.info(f"Trying to load data as prediction {file_paths} ...")
        predictions = [
            prediction
            for json_path in _expand_json_paths(file_paths)
            for prediction in read_json_predictions(json_path).file_predictions_list
        ]
        return (
            classification_system_cls.process(
                ground_truth=GroundTruthBoreholeWithLanguage.from_predictions(predictions=predictions),
                allow_none=True,  # No ground truth label for prediction from extraction
                document_text=document_text,
            ),
            True,
        )
    except Exception:
        pass

    logger.info(f"Fallback, load data as GT (test set) {file_paths} ...")
    gt_boreholes = GroundTruthBoreholeWithLanguage.from_ground_truth(
        ground_truth=_load_ground_truth_dict(file_paths),
    )

    if options.predict_all:
        logger.info("predict_all=True: classifying all descriptions without evaluation.")
        return (
            classification_system_cls.process(ground_truth=gt_boreholes, allow_none=True, document_text=document_text),
            True,
        )

    layer_descriptions_gt = classification_system_cls.process(ground_truth=gt_boreholes, document_text=document_text)
    _, _, layer_descriptions = split_samples(layer_descriptions_gt)

    if not layer_descriptions:
        logger.info("No labeled data found for this classification system. Classifying all descriptions.")
        return (
            classification_system_cls.process(ground_truth=gt_boreholes, allow_none=True, document_text=document_text),
            True,
        )

    return layer_descriptions, False


def _load_document_descriptions(
    file_paths: tuple[Path, ...],
    classification_system_cls: type[ClassificationSystem],
) -> list[LayerInformation]:
    """Extract per-borehole text from one or more PDFs for inference with a document-level system.

    Args:
        file_paths: One or more PDFs, or directories of PDFs.
        classification_system_cls (type[ClassificationSystem]): The document-level classification system.

    Returns:
        list[LayerInformation]: One unlabelled entry per borehole detected across the PDF(s).
    """
    pdf_paths = [pdf for path in file_paths for pdf in ([path] if path.is_file() else sorted(path.glob("*.pdf")))]
    return [
        LayerInformation(
            filename=pdf.name,
            borehole_index=borehole_text.borehole_index,
            layer_index=0,
            language="",  # unused by BertClassifier; only baseline_classifier.py reads this field
            material_description=borehole_text.text,
            class_system=classification_system_cls,
            ground_truth_class=None,
            prediction_class=None,
            llm_reasoning=None,
        )
        for pdf in tqdm(pdf_paths, desc="Extracting text", unit="file")
        for borehole_text in extract_borehole_texts(pdf, pdf.name)
    ]


def _display_paths(paths: tuple[Path, ...]) -> str:
    """Join multiple input paths into a single human-readable string, for tags/reports."""
    return ", ".join(str(p) for p in paths)


def run_classification_predictions(
    file_paths: tuple[Path, ...],
    out_directory: Path,
    out_directory_bedrock: Path,
    options: ClassificationOptions,
) -> tuple[list[LayerInformation] | None, Classifier | None, int]:
    """Load data, run classification, and write predictions.

    This is the core prediction logic, decoupled from tracking and evaluation.

    Args:
        file_paths (tuple[Path, ...]): One or more JSON files (or directories of such files) containing material
            descriptions to classify, merged into a single dataset.
        out_directory (Path): Path to output directory where predictions are written.
        out_directory_bedrock (Path): Path to output directory for Bedrock API files.
        options (ClassificationOptions): Classification run options.

    Returns:
        tuple[list[LayerInformation] | None, Classifier | None, int]: The classified layer descriptions,
            the classifier instance used (or None if no data was found), and the number of
            unique documents processed.
    """
    classifier_type_instance = ClassifierTypes.infer_type(options.classifier_type.lower())
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(
        options.classification_system.lower()
    )
    if classification_system_cls.is_document_level() and _is_pdf_path(file_paths[0]):
        layer_descriptions, is_prediction = _load_document_descriptions(file_paths, classification_system_cls), True
    else:
        document_text = None
        if options.document_texts_paths:
            restrict_to = None
            if classification_system_cls.is_document_level() and not options.predict_all:
                restrict_to = _test_split_filenames(_load_ground_truth_dict(file_paths))
            document_text = _load_document_texts(options.document_texts_paths, restrict_to=restrict_to)
        layer_descriptions, is_prediction = _load_layer_descriptions(
            file_paths, classification_system_cls, options, document_text=document_text
        )

    n_documents = len({layer.filename for layer in layer_descriptions})

    if not layer_descriptions:
        logger.warning("No data to classify.")
        return layer_descriptions, None, n_documents

    classifier = ClassifierFactory.create_classifier(
        classifier_type_instance,
        classification_system_cls,
        options.model_path,
        out_directory_bedrock,
        backbone_path=options.backbone_path,
        tokenizer_path=options.tokenizer_path,
    )
    logger.info(
        f"Classifying layer description into {classification_system_cls.get_name()} classes "
        f"with {classifier.__class__.__name__}"
    )
    layer_descriptions_cls = classifier.classify(layer_descriptions)
    write_predictions(layer_descriptions_cls, str(out_directory / "class_predictions.json"))

    # No layer cls returned, as no metric to compute
    return None if is_prediction else layer_descriptions_cls, classifier, n_documents


@dataclass(kw_only=True)
class ClassificationPipelineRunner(PipelineRunner[_ClassificationResult, ClassificationBenchmarkSummary]):
    """Runs the layer descriptions classification pipeline."""

    file_paths: tuple[Path, ...]
    out_directory: Path
    out_directory_bedrock: Path
    options: ClassificationOptions
    runname: str | None = None
    cleanup_mlflow_tmp: bool = False

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
                "json_file_path": _display_paths(self.file_paths),
                "out_directory": self.out_directory,
            },
            params=None,
        )

    def run_predictions(self, predictions_path_tmp: Path) -> PipelineRunResult[_ClassificationResult]:
        layer_descriptions, classifier, n_documents = run_classification_predictions(
            file_paths=self.file_paths,
            out_directory=self.out_directory,
            out_directory_bedrock=self.out_directory_bedrock,
            options=self.options,
        )

        return PipelineRunResult(
            result=(layer_descriptions, classifier),
            n_documents=n_documents,
        )

    def evaluate(self, run_result: PipelineRunResult[_ClassificationResult]) -> ClassificationBenchmarkSummary | None:
        layer_descriptions, _classifier = run_result.result

        if not layer_descriptions:
            logger.warning("No data to classify. Returning empty summary so parent can still aggregate n_documents.")
            return ClassificationBenchmarkSummary(
                file_path=_display_paths(self.file_paths),
                n_documents=run_result.n_documents,
                classifier_type=self.options.classifier_type,
                model_path=str(self.options.model_path) if self.options.model_path else None,
                classification_system=self.options.classification_system,
                metrics={},
            )

        return evaluate_all_predictions(
            layer_descriptions=layer_descriptions,
            params=BenchmarkParams(
                file_path=_display_paths(self.file_paths),
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
            for file_path in self.file_paths:
                mlflow.log_artifact(str(file_path), "input_data")
            mlflow.log_artifact(f"{self.out_directory}/class_predictions.json", "predictions_json")


@dataclass(kw_only=True)
class ClassificationBenchmarkRunner(MultiBenchmarkRunner[BenchmarkSpec, ClassificationBenchmarkSummary]):
    """Orchestrates multiple classification benchmarks with shared MLflow parent tracking."""

    experiment_name = "Layer descriptions classification"
    input_tag_name = "json_file_path"
    input_path_attr = "file_path"
    aggregate_label = "total/mean"
    _mlflow_use_parent_out_dir = True
    runname = "benchmark"

    options: ClassificationOptions
    out_directory_bedrock: Path

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
            file_paths=(spec.file_path,),
            out_directory=bench_out,
            out_directory_bedrock=bench_out_bedrock,
            options=self.options,
            runname=spec.name,
        ).execute()
