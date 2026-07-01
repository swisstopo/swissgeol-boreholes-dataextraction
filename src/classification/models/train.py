"""Model training module."""

import logging
import os
import shutil
import tempfile
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
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from classification import DATAPATH
from classification.models.config import (
    ExperimentConfig,
    ExperimentDatasetConfig,
    ExperimentHyperparameters,
)
from classification.models.model import BertModel
from classification.utils.datasets import ExistingClassificationSystems
from classification.utils.datasets.classification import (
    ClassificationTask,
    GroundTruthBoreholeWithLanguage,
    LayerInformation,
    split_samples,
)
from classification.utils.file_utils import read_params
from classification.utils.plots import plot_confusion_matrix
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
    config: ExperimentConfig,
    out_directory: Path,
    experiment_name: str = "Bert training",
    run_name: str | None = None,
):
    """Initialise an MLflow run and log experiment metadata.

    Args:
        config (ExperimentConfig): Experiment configuration.
        out_directory (Path): Output directory path, logged as a run tag.
        experiment_name (str): MLflow experiment name. Defaults to "Bert training".
        run_name (str | None): MLflow run name. Defaults to None.
    """
    if not mlflow.active_run():
        if os.getenv("MLFLOW_RUN_ID"):
            # Azure ML pre-allocates a run via MLFLOW_RUN_ID; calling set_experiment()
            # before start_run() causes an experiment-mismatch error, so we skip it.
            mlflow.start_run()
        else:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
    mlflow.set_tag("classification system", config.classification_system)
    mlflow.set_tag("out_directory", str(out_directory))

    with tempfile.TemporaryDirectory() as temp_directory:
        config_path = Path(temp_directory) / "model_config.json"
        config_path.write_text(config.model_dump_json(indent=2))
        mlflow.log_artifact(str(config_path))


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
def train_model(config_file_path: Path, out_directory: Path, model_checkpoint: Path | None):
    """Train a BERT model using the specified datasets and configurations from the YAML config file."""
    model_config = ExperimentConfig.model_validate(read_params(config_file_path))
    classification_system = ExistingClassificationSystems.get_classification_system_type(
        model_config.classification_system
    )

    # If checkpoint model is provided, load from checkpoint output
    work_directory = (
        model_checkpoint.parent
        if model_checkpoint
        else out_directory / classification_system.get_name() / time.strftime("%Y%m%d-%H%M%S")
    )

    if mlflow_tracking:
        logger.info("Logging to MLflow.")
        setup_mlflow_tracking(model_config, work_directory, run_name=model_config.experiment_name)

    # Initialize the model and tokenizer, freeze layers, put in train mode
    logger.info(f"Loading pretrained model from {model_config.model_path}.")
    bert_model = BertModel(model_config.model_path, classification_system)
    bert_model.freeze_all_layers()
    bert_model.unfreeze_list(model_config.unfreeze_layers)
    bert_model.model.train()

    # Load datasets
    logger.info("Loading datasets (transformers library).")
    train_dataset, eval_dataset, test_datasets = setup_data(bert_model, model_config)
    logger.info(
        "Train: %d | Val: %d | Test: %s samples.",
        len(train_dataset),
        len(eval_dataset),
        [len(test_dataset) for test_dataset in test_datasets.values()],
    )

    # Initialize the trainer
    trainer = setup_trainer(bert_model, train_dataset, eval_dataset, model_config, work_directory)

    # Start training
    logger.info("Training ...")
    train_result = trainer.train(resume_from_checkpoint=model_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    logger.info("Evaluation test ...")
    for test_name, test_dataset in test_datasets.items():
        metric_key_prefix = f"test_{test_name}"
        cm_callback = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, ConfusionMatrixCallback))
        cm_callback.current_test_name = test_name
        test_results = trainer.predict(test_dataset, metric_key_prefix=metric_key_prefix)
        trainer.log_metrics(metric_key_prefix, test_results.metrics)
        trainer.save_metrics(metric_key_prefix, test_results.metrics)

    # Save final cleaned version
    logger.info("Saving model head and state ...")
    path_head = trainer.save_fine_tuned_head()

    logger.info("Cleaning checkpoints to save space ...")
    trainer.clean_checkpoints()

    if mlflow_tracking and mlflow.active_run():
        logger.info("Register model and head to MLflow (might take a while) ...")
        mlflow.pytorch.log_model(
            pytorch_model=trainer.model,
            artifact_path="model",
            registered_model_name=model_config.experiment_name,
        )
        mlflow.log_artifacts(path_head, artifact_path="model_head")


def setup_training_args(model_config: ExperimentHyperparameters, out_directory: Path) -> TrainingArguments:
    """Create a TrainingArguments object from the config file.

    Args:
        model_config (ExperimentHyperparameters): The model configuration.
        out_directory (Path): The directory for storing the model.

    Returns:
        TrainingArguments: the training arguments.
    """
    # Read hyperparameters from the config file
    training_args = TrainingArguments(
        output_dir=out_directory,
        logging_dir=out_directory / "logs",
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        num_train_epochs=model_config.num_epochs,
        weight_decay=model_config.weight_decay,
        learning_rate=model_config.learning_rate,
        lr_scheduler_type=model_config.lr_scheduler_type,
        warmup_ratio=model_config.warmup_ratio,
        max_grad_norm=model_config.max_grad_norm,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # metrics logged via MetricsMLflowCallback to avoid Azure ML's 200-param limit
        save_total_limit=2,  # Limit checkpoints to save space, only keep best two
    )
    return training_args


def load_samples_from_set(dataset_cfg: ExperimentDatasetConfig) -> list[LayerInformation]:
    """Load and flatten all labelled layers from a list of dataset configurations.

    Args:
        dataset_cfg (ExperimentDatasetConfig): Configuration specifying ground-truth files
            and the classification system for a single dataset.

    Returns:
        list[LayerInformation]: A flat list of LayerInformation entries from all configured datasets.
    """
    classification_system = ExistingClassificationSystems.get_classification_system_type(
        dataset_cfg.classification_system,
    )
    return [
        sample
        for ground_truth in dataset_cfg.ground_truths
        for sample in classification_system.process(
            ground_truth=GroundTruthBoreholeWithLanguage.from_ground_truth(
                ground_truth=GroundTruth(DATAPATH / ground_truth).ground_truth,
            )
        )
    ]


def setup_data(
    bert_model: BertModel, model_config: ExperimentConfig
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, datasets.Dataset]]:
    """Create tokenized datasets for the train, validation, and test splits.

    The split_samples is deterministic on filename, then there is no overlap between
    train and test slices even when the same files appear in both lists. The train sets
    are merged into a single dataset. The test sets are kept separated for evaluation.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        model_config (ExperimentConfig): The experiment configuration.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset, dict[str, datasets.Dataset]]:
            - Training dataset
            - Validation dataset
            - Test datasets (multiple evaluation possible)
    """
    logger.info("Loading train datasets ...")
    trainval_samples = [
        sample
        for training_set in model_config.training_sets.values()
        for sample in load_samples_from_set(training_set)
    ]
    train_samples, val_samples, _ = split_samples(trainval_samples)
    train_dataset = bert_model.get_tokenized_dataset(train_samples)
    val_dataset = bert_model.get_tokenized_dataset(val_samples)

    logger.info("Loading test datasets ...")
    test_datasets = {}
    for i, (test_name, test_set) in enumerate(model_config.test_sets.items()):
        logger.info(f"[{i + 1} / {len(model_config.test_sets)}] Loading test: {test_name}")
        test_samples = load_samples_from_set(test_set)
        _, _, test_samples = split_samples(test_samples)

        if len(test_samples) == 0:
            logger.warning(f"No samples detected for {test_name=}, omitted")
            continue

        test_datasets[test_name] = bert_model.get_tokenized_dataset(test_samples)
    return train_dataset, val_dataset, test_datasets


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
    labels = trainset["labels"]
    if labels and isinstance(labels[0], list):
        label_counts = Counter(idx for row in labels for idx, val in enumerate(row) if val > 0)
    else:
        label_counts = Counter(labels)
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

    def save_fine_tuned_head(self) -> str:
        """Save only the fine-tuned parameters (requires_grad=True) and the model config.

        Skips frozen backbone weights, keeping checkpoints small and focused on what actually changed.

        Returns:
            str: Folder containing fine-tuned model head
        """
        out_dir = Path(self.args.output_dir) / "model_head"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Gather only trained layers (gradient is available)
        fine_tuned_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        head_state = {k: v.cpu() for k, v in self.model.state_dict().items() if k in fine_tuned_names}

        # Save trained layer and model config
        self.model.config.save_pretrained(out_dir)
        save_file(head_state, out_dir / "model.safetensors")
        return str(out_dir)

    def clean_checkpoints(self) -> None:
        """Clean checkpoints after training."""
        for folder in list(Path(self.args.output_dir).rglob("checkpoint*")):
            if folder.is_dir():
                shutil.rmtree(folder)


def multilabel_confusion_matrix_nxn(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Create nxn confusion matrix for multi-label classification.

    One sample can contribute to multiple cells: cm[i, j] counts samples where label i is true
    and label j is predicted.

    Args:
        labels (np.ndarray): binary indicator matrix of shape (n_samples, n_labels)
        predictions (np.ndarray): binary indicator matrix of shape (n_samples, n_labels)

    Returns:
        np.ndarray: confusion matrix of shape (n_labels, n_labels) where cm[i, j]
            counts samples where label i is true and label j is predicted.
    """
    return (labels.T @ predictions).astype(int)


class ConfusionMatrixCallback(TrainerCallback):
    """Trainer callback to compute and save confusion matrix after evaluation."""

    def __init__(self, id2class_enum: dict, classification_task: ClassificationTask = ClassificationTask.single_label):
        """Initialise the callback.

        Args:
            id2class_enum: Mapping from class index to its enum member.
            classification_task: Type of classification task (default: ClassificationTask.single_label).
        """
        self._id2class_enum = id2class_enum
        self._classification_task = classification_task
        self._sorted_ids = sorted(id2class_enum.keys(), key=lambda i: id2class_enum[i].value)
        self._cm: np.ndarray | None = None
        self.current_test_name: str = "test"

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute per-class and aggregate F1 metrics using sklearn's classification report."""
        logits, labels = eval_pred

        if (
            self._classification_task == ClassificationTask.multi_label
            or self._classification_task == ClassificationTask.rank
        ):
            predictions = (logits > 0).astype(int)
            id_no_prediction = predictions.sum(axis=-1) == 0
            predictions[id_no_prediction, logits[id_no_prediction].argmax(axis=-1)] = 1
            self._cm = multilabel_confusion_matrix_nxn(labels.astype(int), predictions)
        elif (
            self._classification_task == ClassificationTask.single_label
        ):  # single-label: binary label vectors → integer indices (n_samples,)
            predictions = np.zeros_like(logits)
            predictions[range(predictions.shape[0]), logits.argmax(axis=1)] = 1
            self._cm = confusion_matrix(labels.argmax(axis=-1), logits.argmax(axis=-1), labels=self._sorted_ids)
        else:
            raise NotImplementedError(f"Unsupported classification task {self._classification_task}")

        # Drop non existing labels
        (id_keep_col,) = np.nonzero(labels.sum(axis=0) + predictions.sum(axis=0))

        # Use sklearn classification report to get global stats
        report = classification_report(
            labels[:, id_keep_col],
            predictions[:, id_keep_col],
            target_names=[self._id2class_enum[id_keep].name for id_keep in id_keep_col],
            output_dict=True,
        )

        return {
            f"{group_name.replace(' ', '_')}_f1": group_metrics["f1-score"]
            for group_name, group_metrics in report.items()
            if isinstance(group_metrics, dict) and "f1-score" in group_metrics
        }

    def on_predict(self, args, state, control, metrics, **kwargs):
        """Save confusion matrix PNG and per-class metrics CSV to the output directory."""
        csv_path, png_path = plot_confusion_matrix(
            self._cm,
            Path(args.output_dir),
            split=self.current_test_name,
            all_classes=list(self._id2class_enum.values()),
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if "f1" in k and any(tag in k for tag in ["micro", "macro"])}
            )
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(png_path))


class MetricsMLflowCallback(TrainerCallback):
    """Logs step metrics to an active MLflow run.

    Replaces report_to='mlflow' on the Trainer, which dumps all 207 TrainingArguments
    fields as params and exceeds Azure ML MLflow's 200-parameter limit.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log numeric metrics to the active MLflow run at each logging step."""
        if logs and mlflow.active_run():
            mlflow.log_metrics(
                {k: v for k, v in logs.items() if isinstance(v, int | float)},
                step=state.global_step,
            )


def setup_trainer(
    bert_model: BertModel,
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    model_config: ExperimentConfig,
    out_directory: Path,
) -> HeadOnlyTrainer:
    """Create a Trainer object.

    Args:
        bert_model (BertModel): The bert model and tokenizer.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        model_config (ExperimentConfig): The experiment configuration.
        out_directory (Path): The directory for storing the model.

    Returns:
        HeadOnlyTrainer: The trainer object.
    """
    # load the training arguments from the config file
    training_args = setup_training_args(model_config.hyperparameters, out_directory)

    use_class_balancing = model_config.use_class_balancing
    compute_loss_func = None
    if use_class_balancing:
        class_weights = compute_trainset_weights(train_dataset)
        # create the object that will be called to compute the loss function (standard in transformers lib).
        compute_loss_func = WeightedLabelSmoother(class_weights=class_weights)

    cm_callback = ConfusionMatrixCallback(
        id2class_enum=bert_model.id2classEnum,
        classification_task=bert_model.classification_system.classification_task(),
    )
    callbacks = [cm_callback]
    if mlflow_tracking:
        callbacks.append(MetricsMLflowCallback())

    # Create the Trainer object
    trainer = HeadOnlyTrainer(
        model=bert_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=bert_model.tokenizer,
        compute_loss_func=compute_loss_func,
        compute_metrics=cm_callback.compute_metrics,
        callbacks=callbacks,
    )
    return trainer


if __name__ == "__main__":
    # run: fine-tune-bert -cf bert/bert_config_uscs.yml -c models/your_checkpoint_model_folder
    # python -m src.classification.models.train -cf bert/bert_config_color.yml
    train_model()
