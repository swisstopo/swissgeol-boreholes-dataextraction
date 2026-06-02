"""Model training module."""

import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import click
import datasets
import mlflow
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from safetensors.torch import save_file
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from classification import DATAPATH
from classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric
from classification.models.model import BertModel
from classification.utils.datasets import ExistingClassificationSystems
from classification.utils.datasets.classification import GroundTruthBoreholeWithLanguage, split_sets
from classification.utils.file_utils import read_params
from core.benchmark_utils import Metrics
from core.ground_truth import GroundTruth

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
    mlflow.set_tag("json file path", model_config.get("json_file_name"))
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
        help="Path to a local folder containing an existing bert model (e.g. models/../checkpoint-xx).",
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

    # If checkpoint model is provided, load from checkpoint output
    work_directory = (
        model_checkpoint.parent
        if model_checkpoint
        else out_directory / classification_system.get_name() / time.strftime("%Y%m%d-%H%M%S")
    )

    if mlflow_tracking:
        logger.info("Logging to MLflow.")
        setup_mlflow_tracking(model_config, work_directory)

    # Initialize the model and tokenizer, freeze layers, put in train mode
    model_path = model_config["model_path"]
    logger.info(f"Loading pretrained model from {model_path}.")
    bert_model = BertModel(model_path, classification_system)
    bert_model.freeze_all_layers()
    bert_model.unfreeze_list(model_config.get("unfreeze_layers", []))
    bert_model.model.train()

    # Load datasets
    logger.info("Loading datasets (transformers library).")
    train_dataset, eval_dataset, test_dataset = setup_data(bert_model, model_config)
    logger.info(
        "Train: %d | Val: %d | Test: %d samples.",
        len(train_dataset),
        len(eval_dataset),
        len(test_dataset),
    )

    # Initialize the trainer
    trainer = setup_trainer(bert_model, train_dataset, eval_dataset, model_config, work_directory)

    # Start training
    logger.info("Beginning the training.")
    train_result = trainer.train(resume_from_checkpoint=model_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    logger.info("Beginning the test.")
    test_results = trainer.predict(test_dataset)
    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)

    if mlflow_tracking:
        mlflow.log_metrics({k: v for k, v in test_results.metrics.items() if isinstance(v, int | float)})

    # Save final cleaned version
    logger.info("Saving model head and state ...")
    trainer.save_fine_tuned_head()

    logger.info("Cleaning checkpoints (save space) ...")
    trainer.clean_checkpoints()


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


def setup_data(
    bert_model: BertModel, model_config: dict
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Create tokenized datasets for the train, validation, and test splits.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        model_config (dict): The dictionary containing the model configuration.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]: Split datasets.
    """
    file_path = DATAPATH / model_config["json_file_name"]

    classification_system = ExistingClassificationSystems.get_classification_system_type(
        model_config["classification_system"].lower()
    )
    data = classification_system.process(
        ground_truth=GroundTruthBoreholeWithLanguage.from_ground_truth(
            ground_truth=GroundTruth(file_path).ground_truth,
        )
    )

    train_data, val_data, test_data = split_sets(data)
    train_dataset = bert_model.get_tokenized_dataset(train_data)
    val_dataset = bert_model.get_tokenized_dataset(val_data)
    test_dataset = bert_model.get_tokenized_dataset(test_data)
    return train_dataset, val_dataset, test_dataset


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

    return scaled_weights


class HeadOnlyTrainer(Trainer):
    """Trainer that saves only fine-tuned parameters at every checkpoint."""

    def save_fine_tuned_head(self) -> None:
        """Save only the fine-tuned parameters (requires_grad=True) and the model config.

        Skips frozen backbone weights, keeping checkpoints small and focused on what actually changed.
        """
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Gather only trained layers (gradient is available)
        fine_tuned_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        head_state = {k: v.cpu() for k, v in self.model.state_dict().items() if k in fine_tuned_names}

        # Save trained layer and model config
        save_file(head_state, out_dir / "model.safetensors")
        self.model.config.save_pretrained(out_dir)

    def clean_checkpoints(self) -> None:
        """Clean checkpoints after training."""
        for folder in list(Path(self.args.output_dir).rglob("checkpoint*")):
            if folder.is_dir():
                shutil.rmtree(folder)


def setup_trainer(
    bert_model: BertModel,
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    model_config: dict,
    out_directory: Path,
) -> HeadOnlyTrainer:
    """Create a Trainer object.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        model_config (dict): The dictionary containing the model configuration.
        out_directory (Path): The directory for storing the model.

    Returns:
        Trainer: The trainer object.
    """
    # load the training arguments from the config file
    training_args = setup_training_args(model_config, out_directory)

    # Define a custom compute_metrics function
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Function used for evaluating prediction, and logging during the training.

        Note: The metrics are not used to optimize the model during the training, just to evaluate it.
            The model is trained by trying to lower the cross-entropy loss.

        Args:
            eval_pred (EvalPrediction): Object of type EvalPrediction that will be passed to this function.

        Returns:
            dict[str, float]: Dictionary containing all the metrics produced to evaluate the predictions.
        """
        logits, labels = eval_pred
        if labels.ndim == 2:  # multi-label
            predictions = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
            labels = labels.astype(int)
            # compute per-class metrics and micro-average, since sklearn doesn't handle multi-label well
            metric_list = [
                Metrics(
                    tp=int(((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()),
                    fp=int(((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()),
                    fn=int(((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()),
                )
                for i in range(predictions.shape[1])
            ]
            return AllClassificationMetrics.compute_micro_average(metric_list)
        # single-label
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
    trainer = HeadOnlyTrainer(
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
