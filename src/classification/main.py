"""This module contains the main pipeline for the classification of the layer's soil descriptions."""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from utils.file_utils import read_params

from classification import DATAPATH
from classification.classifiers.aws_bedrock_classifier import AWSBedrockClassifier
from classification.classifiers.baseline_classifier import BaselineClassifier
from classification.classifiers.bert_classifier import BertClassifier
from classification.classifiers.classifier_protocol import Classifier
from classification.classifiers.dummy_classifier import DummyClassifier
from classification.evaluation.evaluate import evaluate
from classification.utils.data_loader import LayerInformations, load_data
from classification.utils.data_utils import (
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
    file_path: Path,
    out_directory: Path,
    layer_descriptions: list[LayerInformations],
    classifier: Classifier,
    classification_system: str,
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

    # Log the classification systems used
    mlflow.log_param("classification_systems", classification_system)

    # Log classifier name
    mlflow.log_param("classifier_type", classifier.__class__.__name__)
    if isinstance(classifier, BertClassifier) and os.path.isdir(classifier.model_path):
        mlflow.log_param("model_name", "/".join(classifier.model_path.parts[-2:]))

    # Log model and id, prompt and parameter versions if anthropic model used
    if isinstance(classifier, AWSBedrockClassifier):
        mlflow.log_param("anthropic_model_id", os.environ.get("ANTHROPIC_MODEL_ID"))

        prompt_version = read_params("bedrock/bedrock_config.yml")["prompt_version"]
        if prompt_version:
            mlflow.log_param("anthropic_prompt_version", prompt_version)

        class_param_version = read_params("bedrock/bedrock_config.yml")["uscs_pattern_version"]
        if class_param_version:
            mlflow.log_param("anthropic_class_param_version", class_param_version)

        reasoning_mode = read_params("bedrock/bedrock_config.yml")["reasoning_mode"]
        if reasoning_mode:
            mlflow.log_param("anthropic_reasoning_mode", reasoning_mode)

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
        help="Path to the directory containing subset files (e.g. data/geoquat/validation)."
        " If not provided, the full JSON file is used.",
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
        type=click.Choice(["uscs", "lithology"], case_sensitive=False),
        default="uscs",
        help="The classification system used to classify the data.",
    )(f)
    return f


@click.command()
@common_options
def click_pipeline(
    file_path: Path,
    out_directory: Path,
    out_directory_bedrock: Path,
    file_subset_directory: Path,
    classifier_type: str,
    model_path: Path,
    classification_system: str,
):
    """Run the description classification pipeline."""
    main(
        file_path,
        out_directory,
        out_directory_bedrock,
        file_subset_directory,
        classifier_type,
        model_path,
        classification_system,
    )


def main(
    file_path: Path,
    out_directory: Path,
    out_directory_bedrock: Path,
    file_subset_directory: Path,
    classifier_type: str,
    model_path: Path,
    classification_system: str,
):
    """Main pipeline to classify the layer's soil descriptions.

    Args:
        file_path (Path): Path to the ground truth json file.
        out_directory (Path): Path to output directory
        out_directory_bedrock (Path): Path to output directory for bedrock API files
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        classifier_type (str): The classifier type to use.
        model_path (Path): Path to the trained model.
        classification_system (str): The classification system used to classify the data.
    """
    if classification_system == "lithology" and classifier_type != "dummy":
        raise NotImplementedError(
            "Currently, only the dummy classifier is supported with classification system 'lithology'."
        )

    if mlflow_tracking:
        setup_mlflow_tracking(file_path, out_directory, file_subset_directory)

    logger.info(f"Loading data from {file_path}")
    layer_descriptions = load_data(file_path, file_subset_directory, classification_system)

    if model_path is not None and classifier_type != "bert":
        logger.warning("Model path is only used with classifier 'bert'.")
    if classifier_type == "dummy":
        classifier = DummyClassifier()
    elif classifier_type == "baseline":
        classifier = BaselineClassifier()
    elif classifier_type == "bert":
        classifier = BertClassifier(model_path)
    elif classifier_type == "bedrock":
        classifier = AWSBedrockClassifier(out_directory_bedrock, max_concurrent_calls=1, api_call_delay=0.0)

    # classify
    logger.info(
        f"Classifying layer description into {classification_system} classes with {classifier.__class__.__name__}"
    )
    classifier.classify(layer_descriptions)

    logger.info("Evaluating predictions")
    classification_metrics = evaluate(layer_descriptions)
    logger.info(f"classification metrics: {classification_metrics.to_json()}")
    logger.debug(f"classification metrics per class: {classification_metrics.to_json_per_class()}")

    write_predictions(layer_descriptions, out_directory)
    write_per_language_per_class_predictions(layer_descriptions, classification_metrics, out_directory)

    if mlflow_tracking:
        log_ml_flow_infos(file_path, out_directory, layer_descriptions, classifier, classification_system)

    if mlflow_tracking:
        mlflow.end_run()


if __name__ == "__main__":
    # launch with: python -m src.description_classification.main -f data/geoquat_ground_truth.json
    # or with: boreholes-classify-descriptions  -f data/geoquat_ground_truth.json -s data/geoquat/validation\
    # -c bert -p models/your_best_model_checkpoint_folder
    click_pipeline()
