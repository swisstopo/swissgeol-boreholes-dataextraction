"""Main module."""

import logging
import os
from pathlib import Path

import click
import mlflow

from description_classification import DATAPATH
from description_classification.classifiers.classifiers import Classifier, DummyClassifier
from description_classification.evaluation.evaluate import evaluate
from description_classification.utils.data_loader import load_data

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def setup_mlflow_tracking(
    file_path: Path,
    out_directory: Path,
    experiment_name: str = "Layer descriptions classification",
):
    """Set up MLFlow tracking."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("json file_path", str(file_path))
    mlflow.set_tag("out_directory", str(out_directory))

    # mlflow.log_params(...)

    # repo = pygit2.Repository(".")
    # commit = repo[repo.head.target]
    # mlflow.set_tag("git_branch", repo.head.shorthand)
    # mlflow.set_tag("git_commit_message", commit.message)
    # mlflow.set_tag("git_commit_sha", commit.id)


def log_to_mlflow(classification_metrics):
    """Log metrics to MFlow."""
    with mlflow.start_run():
        mlflow.log_metric("overall_precision", classification_metrics["weighted avg"]["precision"])
        mlflow.log_metric("overall_recall", classification_metrics["weighted avg"]["recall"])
        mlflow.log_metric("overall_f1_score", classification_metrics["weighted avg"]["f1-score"])


def common_options(f):
    """Decorator to add common options to both commands."""
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
    """_summary_.

    Args:
        file_path (Path): _description_
        out_directory (Path): _description_
    """
    if mlflow_tracking:
        setup_mlflow_tracking(file_path, out_directory)

    data_path = "data/geoquat_ground_truth.json"
    descriptions, ground_truth = load_data(data_path)
    classifier: Classifier = DummyClassifier()
    predictions = classifier.classify(descriptions)
    classification_metrics = evaluate(predictions, ground_truth)
    logger.info(f"classification metrics: {classification_metrics}")
    if mlflow_tracking:
        log_to_mlflow(classification_metrics)
        logger.info("Logging metrics to MLFlow")


if __name__ == "__main__":
    click_pipeline()
