"""This module contains the main pipeline for the classification of the layer's soil descriptions."""

from pathlib import Path

import click

from classification import DATAPATH
from classification.evaluation.benchmark.spec import parse_benchmark_spec
from classification.runner import ClassificationBenchmarkRunner, ClassificationOptions, ClassificationPipelineRunner
from core.benchmark_utils import configure_logging


def common_options(f):
    """Decorator to add common options to commands."""
    # input path: can be either a file or dictionary and can ce either be the ground truth descriptions
    # or the predictions
    f = click.option(
        "-f",
        "--file-path",
        "file_paths",
        multiple=True,  # repeatable: -f a.json -f b.json ... merges all of them (each may also be a directory)
        type=click.Path(exists=True, path_type=Path),
        help="Input path to classify. Repeatable: pass one per dataset to merge them into a single run "
        "(e.g. -f zurich_ground_truth.json -f thurgau_ground_truth.json). Each can be a ground truth "
        "JSON, a predictions JSON, or a directory of such files. For document-level systems (e.g. "
        "borehole_type), a PDF file or a directory of PDFs instead.",
    )(f)
    f = click.option(
        "-dt",
        "--document-texts",
        "document_texts",
        multiple=True,  # repeatable, same merge semantics as -f
        type=click.Path(exists=True, path_type=Path),
        help="For document-level systems (e.g. borehole_type) when -f is ground truth JSON rather than "
        "PDFs: one or more sources of filename -> text, merged together. Each can be a precomputed "
        "filename -> text JSON file/directory (fast, e.g. training's cached *_filtered_text.json), or a "
        "PDF file/directory of PDFs, extracted live via the same pipeline used for raw-PDF inference "
        "-- mix both as needed. Ignored for layer-level systems and for PDF input via -f.",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output_description_classification",
        help="Path to the output directory.",
    )(f)
    f = click.option(
        "-ob",
        "--out-directory-bedrock",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output_description_classification_bedrock",
        help="Path to the output directory for bedrock files.",
    )(f)
    f = click.option(
        "-c",
        "--classifier-type",
        type=click.Choice(["dummy", "baseline", "bert", "bedrock"], case_sensitive=False),
        default="dummy",
        help="Classifier to use for description classification. Choose from 'dummy', 'baseline', 'bert' or 'bedrock'.",
    )(f)
    f = click.option(
        "-p",
        "--model-path",
        type=str,
        default=None,
        help="Local path to the model directory or a HuggingFace model ID (e.g. 'swissgeol/en_main'). "
        "For split models this is the head directory.",
    )(f)
    f = click.option(
        "-b",
        "--backbone-path",
        type=click.Path(path_type=Path),
        default=None,
        help="Path to backbone.safetensors for split-model loading. "
        "When provided, --model-path is the head directory.",
    )(f)
    f = click.option(
        "-t",
        "--tokenizer-path",
        type=click.Path(path_type=Path),
        default=None,
        help="Directory containing tokenizer files. When omitted, falls back to --model-path.",
    )(f)
    f = click.option(
        "-cs",
        "--classification-system",
        type=click.Choice(
            [
                "accessory_components",
                "alteration_degree_consolidated",
                "alteration_degree_unconsolidated",
                "borehole_type",
                "cementation",
                "color_consolidated",
                "color_unconsolidated",
                "debris",
                "en_main",
                "en_secondary",
                "grain_angularity",
                "grain_shape",
                "lithology",
                "mineral_components",
                "organic_components",
                "uscs",
            ],
            case_sensitive=False,
        ),
        default="uscs",
        help="The classification system used to classify the data.",
    )(f)
    f = click.option(
        "-r",
        "--resume",
        is_flag=True,
        default=False,
        help="Whether to resume previous run. Defaults to False.",
    )(f)
    f = click.option(
        "--predict-all",
        is_flag=True,
        default=False,
        help="Classify every layer description regardless of existing labels. "
        "Skips train/test splitting and evaluation. Use when generating labels for a full dataset.",
    )(f)

    return f


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    help="Repeatable benchmark spec: '<name>:<input_path>'. If provided, runs multiple benchmarks in one execution.",
)
@common_options
def click_pipeline(
    file_paths: tuple[Path, ...],
    document_texts: tuple[Path, ...],
    out_directory: Path,
    out_directory_bedrock: Path,
    classifier_type: str,
    model_path: str | None,
    backbone_path: Path | None,
    tokenizer_path: Path | None,
    classification_system: str,
    resume: bool,
    predict_all: bool,
    benchmarks: tuple[str, ...] = (),
):
    """Command line interface for the classification pipeline (single or multi-benchmark)."""
    configure_logging()
    # --- Multi-benchmark mode ---
    opts = ClassificationOptions(
        classifier_type=classifier_type,
        model_path=model_path,
        backbone_path=backbone_path,
        tokenizer_path=tokenizer_path,
        classification_system=classification_system,
        predict_all=predict_all,
        document_texts_paths=document_texts,
    )

    if benchmarks:
        specs = [parse_benchmark_spec(b) for b in benchmarks]
        multi_root = out_directory / "multi"
        multi_bedrock_root = out_directory_bedrock / "multi"
        ClassificationBenchmarkRunner(
            benchmarks=specs,
            multi_root=multi_root,
            resume=resume,
            options=opts,
            out_directory_bedrock=multi_bedrock_root,
        ).run()
        return

    # --- Single-benchmark mode ---
    if not file_paths:
        raise click.BadParameter("Missing -f/--file-path. Provide at least one, or use one or more --benchmark specs.")

    ClassificationPipelineRunner(
        predictions_path=out_directory / "class_predictions.json",
        file_paths=file_paths,
        out_directory=out_directory,
        out_directory_bedrock=out_directory_bedrock,
        options=opts,
    ).execute()


if __name__ == "__main__":
    click_pipeline()
