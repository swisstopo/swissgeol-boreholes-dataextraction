"""Model training module."""

import logging
import os
import time
from collections import Counter
from pathlib import Path

import click
import datasets
import mlflow
import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from classification import DATAPATH
from classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric
from classification.models.model import BertModel
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.data_loader import prepare_classification_data
from classification.utils.file_utils import read_params

if __name__ == "__main__":
    # Only configure logging if this script is run directly (e.g. training pipeline entrypoint)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"


class WeightedLabelSmoother:
    """Label Smoothing with optional per-class weighting for classification tasks.

    Acts as a loss function when called. It is the standard way of doing in the transformers librairy.
    Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L539.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __init__(self, class_weights: torch.Tensor = None):
        """Initialize the object.

        Args:
            class_weights (torch.Tensor, optional): A 1D tensor of shape (num_classes,) with per-class weights.
        """
        self.class_weights = class_weights  # Tensor of shape [num_classes]

    def __call__(
        self, model_output: SequenceClassifierOutput, labels: torch.Tensor, num_items_in_batch: torch.Tensor = None
    ) -> torch.Tensor:
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

        log_probs = -nn.functional.log_softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)  # shape: (batch_size, seq_len, 1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case
        safe_labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=safe_labels)  # shape: (batch_size, seq_len, 1)

        # New: Apply per-class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            per_token_weights = weights[safe_labels.squeeze(-1)]  # shape: (batch_size, seq_len)
            per_token_weights = per_token_weights.unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
            nll_loss = nll_loss * per_token_weights

        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)  # (batch_size, seq_len, 1)

        nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll = nll_loss.sum() / num_active_elements
        smooth = smoothed_loss.sum() / (num_active_elements * log_probs.size(-1))
        return (1 - self.epsilon) * nll + self.epsilon * smooth


def setup_mlflow_tracking(
    model_config: dict,
    out_directory: Path,
    experiment_name: str = "Bert training",
):
    """Set up MLFlow tracking."""
    if mlflow.active_run():
        mlflow.end_run()  # Ensure the previous run is closed
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("classification system", str(model_config["classification_system"]))
    json_path = model_config.get("json_file_name")
    json_path = json_path if json_path else model_config.get("train_subset").split("/")[0]
    mlflow.set_tag("json file path", json_path)
    mlflow.set_tag("out_directory", str(out_directory))
    mlflow.log_params(model_config)


def common_options(f):
    """Decorator to add common options to commands."""
    f = click.option(
        "-cf",
        "--config-file-path",
        required=True,
        type=str,
        help="Name (not path) of the configuration yml file inside the `config` folder.",
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
def train_model(config_file_path: Path, out_directory: Path, model_checkpoint: Path):
    """Train a BERT model using the specified datasets and configurations from the YAML config file."""
    model_config = read_params(config_file_path)
    classification_system = ExistingClassificationSystems.get_classification_system_type(
        model_config["classification_system"].lower()
    )

    out_directory = out_directory / classification_system.get_name() / time.strftime("%Y%m%d-%H%M%S")

    if mlflow_tracking:
        logger.info("Logging to MLflow.")
        setup_mlflow_tracking(model_config, out_directory)

    # Initialize the model and tokenizer, freeze layers, put in train mode

    model_path = model_config["model_path"] if model_checkpoint is None else model_checkpoint
    logger.info(f"Loading pretrained model from {model_path}.")
    bert_model = BertModel(model_path, classification_system)
    bert_model.freeze_all_layers()
    bert_model.unfreeze_list(model_config.get("unfreeze_layers", []))
    bert_model.model.train()

    # Initialize the trainer
    trainer = setup_trainer(bert_model, model_config, out_directory)

    # Start training
    logger.info("Beginning the training.")
    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def setup_training_args(model_config: dict, out_directory: Path) -> TrainingArguments:
    """Create a TrainingArgument object from the config file.

    Args:
        model_config (dict): The dictionary containing the model configuration.
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


def setup_data(bert_model: BertModel, model_config: dict) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Create tokenized datasets for the train and evaluation parts.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        model_config (dict): The dictionary containing the model configuration.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: the training arguments.
    """
    if model_config["classification_system"] == "uscs":
        # the data is not stored the same way for uscs and lithology. Currently the reports names enumerated in the
        # json file only locally exists for uscs.
        # Once all of the files are available, we will be able to use the code without the need for this
        # if-else block.
        train_file_path = DATAPATH / model_config["json_file_name"]
        train_subset = DATAPATH / model_config["train_subset"]
        eval_file_path = DATAPATH / model_config["json_file_name"]
        eval_subset = DATAPATH / model_config["eval_subset"]
    elif model_config["classification_system"] == "lithology":
        train_file_path = DATAPATH / model_config["train_subset"]
        train_subset = None
        eval_file_path = DATAPATH / model_config["eval_subset"]
        eval_subset = None

    classification_system = ExistingClassificationSystems.get_classification_system_type(
        model_config["classification_system"].lower()
    )
    train_data = prepare_classification_data(
        train_file_path,
        ground_truth_path=None,
        file_subset_directory=train_subset,
        classification_system=classification_system,
    )
    train_dataset = bert_model.get_tokenized_dataset(train_data)
    eval_data = prepare_classification_data(
        eval_file_path,
        ground_truth_path=None,
        file_subset_directory=eval_subset,
        classification_system=classification_system,
    )
    eval_dataset = bert_model.get_tokenized_dataset(eval_data)
    return train_dataset, eval_dataset


def compute_trainset_weights(
    trainset: datasets.Dataset, min_scale: float = 0.5, max_scale: float = 2.0
) -> torch.Tensor:
    """Computes normalized inverse-frequency class weights.

    Args:
        trainset (datasets.Dataset): the dataset to infer the weights from.
        min_scale (float): Minimum weight value after scaling.
        max_scale (float): Maximum weight value after scaling.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) with scaled weights.
    """
    label_counts = Counter(trainset["label"])
    num_classes = max(label_counts.keys()) + 1  # class index starts at 0

    # Compute raw inverse-frequency weights
    raw_weights = torch.tensor(
        [1.0 / label_counts[i] if i in label_counts else 0.0 for i in range(num_classes)], dtype=torch.float32
    )

    # Scale to desired range [min_scale, max_scale]
    nonzero = raw_weights[raw_weights > 0]
    if len(nonzero) > 0 and nonzero.max() != nonzero.min():
        min_w, max_w = nonzero.min(), nonzero.max()
        scaled_weights = (raw_weights - min_w) / (max_w - min_w)  # normalize to [0, 1]
        scaled_weights = min_scale + (max_scale - min_scale) * scaled_weights  # scale to [min_scale, max_scale]
    else:
        scaled_weights = torch.ones_like(raw_weights)  # fallback: uniform weights

    # OR other method
    # scaled_weights = torch.tensor([
    #     1.0 / np.log(1 + label_counts.get(i, 1)) for i in range(num_classes)
    # ])
    return scaled_weights


def setup_trainer(bert_model: BertModel, model_config: dict, out_directory: Path) -> Trainer:
    """Create a Trainer object.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        model_config (dict): The dictionary containing the model configuration.
        out_directory (Path): The directory for storing the model.

    Returns:
        Trainer: The trainer obect.
    """
    # load the training arguments from the config file
    training_args = setup_training_args(model_config, out_directory)

    # Load datasets
    logger.info("Loading datasets (transformers librairy).")
    train_dataset, eval_dataset = setup_data(bert_model, model_config)

    # Define a custom compute_metrics function
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Function used for evaluating prediction, and logging during the training.

        Note: The metrics are not used to optimize the model during the training, just to evaluate it.
            The model is trained by trying to lower the cross-entropy loss.

        Args:
            eval_pred (EvalPrediction): Object of type EvalPrediction that will be passt to this function.

        Returns:
            dict[str, float]: Dictionary containing all the metrics procduced to evaluate the predictions.
        """
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        metrics = per_class_metric(predictions, labels)
        return AllClassificationMetrics.compute_micro_average(metrics.values())

    use_class_balacing = model_config.get("use_class_balancing", "false").lower() == "true"
    compute_loss_func = None
    if use_class_balacing:
        class_weights = compute_trainset_weights(train_dataset)
        # create the object that will be called to compute the loss function (standard in transformers lib).
        compute_loss_func = WeightedLabelSmoother(class_weights=class_weights)

    # Create the Trainer object
    trainer = Trainer(
        model=bert_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=bert_model.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=bert_model.tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
    )
    return trainer


if __name__ == "__main__":
    # run: fine-tune-bert -cf bert_config_uscs.yml -c models/your_chekpoint_model_folder
    train_model()
