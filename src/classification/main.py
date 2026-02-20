"""This module contains the main pipeline for the classification of the layer's soil descriptions."""

from pathlib import Path

import click

from classification import DATAPATH
from classification.evaluation.benchmark.spec import parse_benchmark_spec
from classification.runner import start_multi_benchmark, start_pipeline
from core.benchmark_utils import configure_logging


def common_options(f):
    """Decorator to add common options to commands."""
    f = click.option(
        "-f",
        "--file-path",
        required=False,  # allow multi-benchmark mode
        type=click.Path(exists=True, path_type=Path),
        help="Path to the json file containing the material descriptions to classify.",
    )(f)
    f = click.option(
        "-g",
        "--ground-truth-path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to the ground truth file, if different from file_path.",
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
        "-s",
        "--file-subset-directory",
        type=click.Path(path_type=Path),
        default=None,
        help="Path to the directory containing subset files (e.g. data/geoquat/validation). "
        "If not provided, the full JSON file is used.",
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
        type=click.Path(path_type=Path),
        default=None,
        help="Path to the local trained model.",
    )(f)
    f = click.option(
        "-cs",
        "--classification-system",
        type=click.Choice(["uscs", "lithology", "en_main"], case_sensitive=False),
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

    return f


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    help="Repeatable benchmark spec: '<name>:<file_path>:<file_subset_directory>' "
    "or '<name>:<file_path>:<file_subset_directory>:<ground_truth_path>'. "
    "If provided, runs multiple benchmarks in one execution.",
)
@common_options
def click_pipeline(
    file_path: Path | None,
    ground_truth_path: Path | None,
    out_directory: Path,
    out_directory_bedrock: Path,
    file_subset_directory: Path | None,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    resume: bool,
    benchmarks: tuple[str, ...] = (),
):
    """Command line interface for the classification pipeline (single or multi-benchmark)."""
    configure_logging()
    # --- Multi-benchmark mode ---
    if benchmarks:
        specs = [parse_benchmark_spec(b) for b in benchmarks]
        start_multi_benchmark(
            benchmarks=specs,
            out_directory=out_directory,
            out_directory_bedrock=out_directory_bedrock,
            classifier_type=classifier_type,
            model_path=model_path,
            classification_system=classification_system,
            resume=resume,
        )
        return

    # --- Single-benchmark mode ---
    if file_path is None:
        raise click.BadParameter("Missing -f/--file-path. Provide it, or use one or more --benchmark specs.")

    start_pipeline(
        file_path=file_path,
        ground_truth_path=ground_truth_path,
        out_directory=out_directory,
        out_directory_bedrock=out_directory_bedrock,
        predictions_path=out_directory / "class_predictions.json",
        file_subset_directory=file_subset_directory,
        classifier_type=classifier_type,
        model_path=model_path,
        classification_system=classification_system,
    )


if __name__ == "__main__":
    click_pipeline()
