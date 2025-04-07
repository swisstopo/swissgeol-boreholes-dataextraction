"""This module contains the main pipeline for the classification of the layer's soil descriptions."""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from stratigraphy.util.util import read_params

from description_classification import DATAPATH
from description_classification.classifiers.classifiers import BaselineClassifier, Classifier
from description_classification.evaluation.evaluate import evaluate
from description_classification.utils.data_loader import LayerInformations, load_data
from description_classification.utils.data_utils import (
    get_data_class_count,
    get_data_language_count,
    write_per_language_per_class_predictions,
    write_predictions,
)

load_dotenv()
classification_params = read_params("classification_params.yml")

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def setup_mlflow_tracking(
    file_path: Path,
    out_directory: Path,
    file_subset_directory: Path,
    experiment_name: str = "Layer descriptions classification",
):
    """Set up MLFlow tracking."""
    if mlflow.active_run():
        mlflow.end_run()  # Ensure the previous run is closed
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("json file_path", str(file_path))
    mlflow.set_tag("out_directory", str(out_directory))
    mlflow.set_tag("file_subset_directory", str(file_subset_directory))


def log_ml_flow_infos(
    file_path: Path, out_directory: Path, layer_descriptions: list[LayerInformations], classifier: Classifier
):
    """Logs informations to mlflow, such as the number of sample, laguage distribution, classifier type and data."""
    # Log dataset statistics
    mlflow.log_param("dataset_size", len(layer_descriptions))

    # Log language distribution
    for language, count in get_data_language_count(layer_descriptions).items():
        mlflow.log_param(f"language_{language}_count", count)

    # Log class distribution
    for class_, count in get_data_class_count(layer_descriptions).items():
        mlflow.log_param(f"class_{class_}_count", count)

    # Log model name
    mlflow.log_param("classifier_type", classifier.__class__.__name__)

    # Log input data and output predictions
    mlflow.log_artifact(str(file_path), "input_data")
    mlflow.log_artifact(f"{out_directory}/uscs_class_predictions.json", "predictions_json")

    pred_dir = os.path.join(out_directory, "predictions_per_ground_truth_class")
    for language in ["global", *classification_params["supported_language"]]:
        language_dir = os.path.join(pred_dir, language)
        mlflow.log_artifact(
            os.path.join(language_dir, "_overview.json"), f"predictions_per_ground_truth_class_json/{language}"
        )
        for file in os.listdir(language_dir):
            file_path = os.path.join(language_dir, file)
            mlflow.log_artifact(file_path, f"predictions_per_ground_truth_class_json/{language}")


def common_options(f):
    """Decorator to add common options to commands."""
    f = click.option(
        "-f",
        "--file-path",
        required=True,
        type=click.Path(exists=True, path_type=Path),
        help="Path to the json file.",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default=DATAPATH / "output_description_classification",
        help="Path to the output directory.",
    )(f)
    f = click.option(
        "-s",
        "--file-subset-directory",
        type=click.Path(path_type=Path),
        default=None,
        help="Path to the directory containing subset files (e.g. data/geoquat/train)."
        " If not provided, the full JSON file is used.",
    )(f)
    return f


@click.command()
@common_options
def click_pipeline(file_path: Path, out_directory: Path, file_subset_directory: Path):
    """Run the description classification pipeline."""
    main(file_path, out_directory, file_subset_directory)


def main(file_path: Path, out_directory: Path, file_subset_directory: Path):
    """Main pipeline to classify the layer's soil descriptions.

    Args:
        file_path (Path): Path to the ground truth json file.
        out_directory (Path): Path to output directory
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
    """
    if mlflow_tracking:
        setup_mlflow_tracking(file_path, out_directory, file_subset_directory)

    logger.info(f"Loading data from {file_path}")
    layer_descriptions = load_data(file_path, file_subset_directory)

    classifier: Classifier = BaselineClassifier()
    logger.info(f"Classifying layer description with {classifier.__class__.__name__}")
    classifier.classify(layer_descriptions)

    write_predictions(layer_descriptions, out_directory)
    write_per_language_per_class_predictions(layer_descriptions, out_directory)

    if mlflow_tracking:
        log_ml_flow_infos(file_path, out_directory, layer_descriptions, classifier)

    logger.info("Evaluating predictions")
    classification_metrics = evaluate(layer_descriptions)
    logger.info(f"classification metrics: {classification_metrics.to_json()}")
    logger.debug(f"classification metrics per class: {classification_metrics.to_json_per_class()}")

    if mlflow_tracking:
        mlflow.end_run()


if __name__ == "__main__":
    # launch with: python -m src.description_classification.main -f data/geoquat_ground_truth.json
    # or with: boreholes-classify-descriptions  -f data/geoquat_ground_truth.json
    click_pipeline()
