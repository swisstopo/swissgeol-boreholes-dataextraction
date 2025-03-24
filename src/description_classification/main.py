"""This module contains the main pipeline for the classification of the layer's soil descriptions."""

import logging
import os
from pathlib import Path

import click
import mlflow
from dotenv import load_dotenv

from description_classification import DATAPATH
from description_classification.classifiers.classifiers import BaselineClassifier, Classifier
from description_classification.evaluation.evaluate import evaluate
from description_classification.utils.data_loader import load_data, write_predictions

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def setup_mlflow_tracking(
    file_path: Path,
    out_directory: Path,
    experiment_name: str = "Layer descriptions classification",
):
    """Set up MLFlow tracking."""
    if mlflow.active_run():
        mlflow.end_run()  # Ensure the previous run is closed
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("json file_path", str(file_path))
    mlflow.set_tag("out_directory", str(out_directory))


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
    return f


@click.command()
@common_options
def click_pipeline(
    file_path: Path,
    out_directory: Path,
):
    """Run the description classification pipeline."""
    main(file_path, out_directory)


def main(file_path: Path, out_directory: Path):
    """Main pipeline to classify the layer's soil descriptions.

    Args:
        file_path (Path): Path to the ground truth json file.
        out_directory (Path): Path to output directory
    """
    if mlflow_tracking:
        setup_mlflow_tracking(file_path, out_directory)

    data_path = Path("data/geoquat_ground_truth.json")
    logger.info(f"Loading data from {data_path}")
    layer_descriptions = load_data(data_path)

    # classifier: Classifier = DummyClassifier()
    classifier: Classifier = BaselineClassifier()
    logger.info(f"Classifying layer description with {classifier.__class__.__name__}")
    classifier.classify(layer_descriptions)

    write_predictions(layer_descriptions, out_directory)

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
