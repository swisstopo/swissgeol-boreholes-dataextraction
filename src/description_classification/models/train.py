"""Model training module."""

import logging
import os
import time
from pathlib import Path

import click
import datasets
import mlflow
from dotenv import load_dotenv
from stratigraphy.util.util import read_params
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from description_classification import DATAPATH
from description_classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric
from description_classification.models.model import BertModel
from description_classification.utils.data_loader import load_data

if __name__ == "__main__":
    # Only configure logging if this script is run directly (e.g., training pipeline entrypoint)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

model_config = read_params("bert_config.yml")


def setup_mlflow_tracking(
    file_path: Path,
    out_directory: Path,
    experiment_name: str = "Bert training",
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
        "-c",
        "--model-checkpoint",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to a local folder containing an existing bert model (e.g. models/your_model_folder).",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default="models",
        help="Path to the output directory.",
    )(f)
    return f


@click.command()
@common_options
def train_model(file_path: Path, out_directory: Path, model_checkpoint: Path):
    """Train a BERT model using the specified datasets and configurations from the YAML config file."""
    out_directory = out_directory / time.strftime("%Y%m%d-%H%M%S")

    if mlflow_tracking:
        logger.info("Logging to MLflow.")
        setup_mlflow_tracking(file_path, out_directory)

    # Initialize the model and tokenizer, freeze layers, put in train mode
    model_path = model_config["model_path"] if model_checkpoint is None else model_checkpoint
    logger.info(f"Loading pretrained model from {model_path}.")
    bert_model = BertModel(model_path)
    bert_model.freeze_layers_except_pooler_and_classifier()
    bert_model.model.train()

    # Initialize the trainer
    trainer = setup_trainer(bert_model, file_path, out_directory)

    # Start training
    logger.info("Beginning the training.")
    train_result = trainer.train(resume_from_checkpoint=model_checkpoint)

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def setup_training_args(out_directory: Path) -> TrainingArguments:
    """Create a TrainingArgument object from the config file.

    Args:
        out_directory (Path): The directory for storing the model.

    Returns:
        TrainingArgument: the training arguments.
    """
    report_to = "mlflow" if mlflow_tracking else "none"
    # Read hyperparameters from the config file
    training_args = TrainingArguments(
        output_dir=out_directory,
        logging_dir=out_directory / "logs",
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        num_train_epochs=model_config["num_epochs"],
        weight_decay=float(model_config["weight_decay"]),
        learning_rate=float(model_config["learning_rate"]),
        lr_scheduler_type=model_config["lr_scheduler_type"],
        warmup_ratio=float(model_config["warmup_ratio"]),
        max_grad_norm=float(model_config["max_grad_norm"]),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=report_to,
        save_total_limit=2,  # Limit checkpoints to save space, only keep best two
    )
    return training_args


def setup_data(bert_model: BertModel, file_path: Path) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Create tokenized datasets for the train and evaluation parts.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        file_path (Path) : The path to the json file with the description and labels.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: the training arguments.
    """
    train_subset = model_config["train_subset"]
    eval_subset = model_config["eval_subset"]

    train_data = load_data(file_path, file_subset_directory=DATAPATH / train_subset)
    train_dataset = bert_model.get_tokenized_dataset(train_data)
    eval_data = load_data(file_path, file_subset_directory=DATAPATH / eval_subset)
    eval_dataset = bert_model.get_tokenized_dataset(eval_data)
    return train_dataset, eval_dataset


def setup_trainer(bert_model: BertModel, file_path: Path, out_directory: Path) -> Trainer:
    """Create a Trainer object.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        file_path (Path): The path to the json file with the description and labels.
        out_directory (Path): The directory for storing the model.

    Returns:
        Trainer: The trainer obect.
    """
    # load the training arguments from the config file
    training_args = setup_training_args(out_directory)

    # Load datasets
    logger.info("Loading datasets (transformers librairy).")
    train_dataset, eval_dataset = setup_data(bert_model, file_path)

    # Define a custom compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        metrics = per_class_metric(predictions, labels)
        return AllClassificationMetrics.compute_micro_average(metrics.values())

    # Create the Trainer object
    trainer = Trainer(
        model=bert_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=bert_model.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=bert_model.tokenizer),
        compute_metrics=compute_metrics,  # Define your custom metric function if needed
    )
    return trainer


if __name__ == "__main__":
    # run: fine-tune-bert -f data/geoquat_ground_truth.json -c models/your_chekpoint_model_folder
    train_model()
