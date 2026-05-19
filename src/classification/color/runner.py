"""Color classification training runner."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import datasets as hf_datasets
import torch
from dotenv import load_dotenv
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from classification.color.dataset_config import ColorDatasetConfig, load_splits
from classification.data_loader.color_data_loader import COLORS, BoreholeDataset
from classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric
from classification.models.train import WeightedLabelSmoother, compute_trainset_weights
from core.mlflow_tracking import mlflow
from core.mlflow_utils import setup_mlflow_tracking

load_dotenv()
logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"


class _BestFinetuneCheckpointer(TrainerCallback):
    """Saves only classifier + pooler weights after each eval; reloads the best at training end.

    Replaces Trainer's built-in checkpoint mechanism so no full BERT weights (~440 MB)
    are written to disk during training — only the fine-tuned layers (~54 KB).
    """

    def __init__(self, model, checkpoint_dir: Path) -> None:
        self._model = model
        self._checkpoint_dir = checkpoint_dir
        self._best_metric: float = float("-inf")
        self._best_path: Path | None = None

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs
    ) -> None:
        val_accuracy = metrics.get("val_accuracy", float("-inf"))
        if val_accuracy <= self._best_metric:
            return
        self._best_metric = val_accuracy
        new_path = self._checkpoint_dir / f"best_finetuned_epoch{round(state.epoch)}.pt"
        torch.save(
            {"classifier": self._model.classifier.state_dict(), "pooler": self._model.bert.pooler.state_dict()},
            new_path,
        )
        if self._best_path and self._best_path != new_path and self._best_path.exists():
            self._best_path.unlink()
        self._best_path = new_path
        logger.info("New best val_accuracy=%.4f — fine-tuned layers saved.", val_accuracy)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._best_path and self._best_path.exists():
            best = torch.load(self._best_path, weights_only=True)
            self._model.classifier.load_state_dict(best["classifier"])
            self._model.bert.pooler.load_state_dict(best["pooler"])
            logger.info(
                "Restored best fine-tuned layers from %s (val_accuracy=%.4f).", self._best_path.name, self._best_metric
            )


def _make_weighted_sampler(dataset: hf_datasets.Dataset) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples rare color classes.

    Each sample's weight is the inverse frequency of its class, so every color
    appears roughly equally often across training batches regardless of how
    imbalanced the raw dataset is.
    """
    labels = dataset["label"]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.float)
    return WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


class _BalancedTrainer(Trainer):
    """Trainer that replaces the default sequential sampler with a class-balanced one."""

    def __init__(self, *args, train_sampler: WeightedRandomSampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, dataset=None) -> WeightedRandomSampler:
        return self._train_sampler


@dataclass
class ColorTrainingResult:
    """Outcome of a single color training run.

    Attributes:
        config:    The dataset configuration that was used.
        model_dir: Directory where the best checkpoint was saved.
        metrics:   Final evaluation metrics on the test set (accuracy, micro F1/precision/recall).
    """

    config: ColorDatasetConfig
    model_dir: Path
    metrics: dict[str, float]


def _to_hf_dataset(dataset: BoreholeDataset, tokenizer, max_length: int = 512) -> hf_datasets.Dataset:
    """Convert a BoreholeDataset to a tokenized HuggingFace Dataset.

    Drops samples whose color one-hot vector is all zeros (no color label).
    The one-hot vector is collapsed to a scalar label index via argmax.
    """
    texts, labels = [], []
    for sample in dataset:
        color_vec = sample["color"]
        if not any(color_vec):
            continue
        texts.append(sample["material_description"])
        labels.append(color_vec.index(1))

    hf_ds = hf_datasets.Dataset.from_dict({"text": texts, "label": labels})

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    return hf_ds.map(_tokenize, batched=True, remove_columns=["text"])


@dataclass(kw_only=True)
class ColorTrainingRunner:
    """Trains a BERT model to predict color from borehole layer descriptions.

    Follows the same pattern as classification/models/train.py but uses
    ColorDatasetConfig + BoreholeDataset instead of the ClassificationSystem pipeline.

    Attributes:
        config: Dataset split configuration (which slices to train/test on).
        model_path: Path to a local BERT checkpoint or a HuggingFace model ID.
        out_directory: Root directory for model checkpoints and logs.
        batch_size: Per-device batch size for training and evaluation.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularisation weight.
        warmup_ratio: Fraction of steps used for LR warm-up.
        lr_scheduler_type: LR schedule type (e.g. "cosine", "linear").
        max_grad_norm: Gradient clipping threshold.
        use_class_balancing: If True, weights loss by inverse class frequency.
    """

    config: ColorDatasetConfig
    model_path: str | Path
    out_directory: Path

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 5.0
    max_length: int = 128
    use_class_balancing: bool = False

    def run(self) -> ColorTrainingResult:
        """Train, evaluate, and save the model.

        Returns:
            ColorTrainingResult with the saved model directory and final test metrics.
        """
        run_dir = self.out_directory / self.config.name / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        if mlflow_tracking:
            setup_mlflow_tracking(
                run_id=None,
                experiment_name="Color classification",
                runname=self.config.name,
                params={
                    "train_slices": str(self.config.train_slices),
                    "test_slices": str(self.config.test_slices),
                    "model_path": str(self.model_path),
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "warmup_ratio": self.warmup_ratio,
                    "lr_scheduler_type": self.lr_scheduler_type,
                    "max_grad_norm": self.max_grad_norm,
                    "max_length": self.max_length,
                    "use_class_balancing": self.use_class_balancing,
                },
            )

        # 1. Load splits
        logger.info("Loading dataset splits for config '%s'.", self.config.name)
        train_ds, val_ds, test_ds = load_splits(self.config)

        # 2. Tokenizer + model
        id2label = dict(enumerate(COLORS))
        label2id = {c: i for i, c in enumerate(COLORS)}

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=len(COLORS),
            id2label=id2label,
            label2id=label2id,
        )

        # Freeze backbone — classifier + pooler are trained (mirrors lithology config)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier.") or name.startswith("bert.pooler.")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Backbone frozen. Trainable parameters: %d (classifier + pooler).", trainable)
        model.train()

        # 3. Tokenize
        logger.info("Tokenizing splits (max_length=%d).", self.max_length)
        train_hf = _to_hf_dataset(train_ds, tokenizer, self.max_length)
        val_hf = _to_hf_dataset(val_ds, tokenizer, self.max_length)
        test_hf = _to_hf_dataset(test_ds, tokenizer, self.max_length)
        logger.info("Train: %d | Val: %d | Test: %d samples.", len(train_hf), len(val_hf), len(test_hf))

        if len(train_hf) == 0:
            logger.error("Training set is empty — no samples with color labels found. Aborting.")
            return ColorTrainingResult(config=self.config, model_dir=run_dir, metrics={})

        # Small training sample evaluated each epoch alongside val — shows training vs val accuracy
        train_sample_hf = train_hf.select(range(min(512, len(train_hf))))

        # 4. Training arguments
        training_args = TrainingArguments(
            output_dir=run_dir,
            logging_dir=run_dir / "logs",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            report_to="mlflow" if mlflow_tracking else "none",
        )

        # 5. Optional class balancing
        compute_loss_func = None
        if self.use_class_balancing:
            class_weights = compute_trainset_weights(train_hf)
            compute_loss_func = WeightedLabelSmoother(class_weights=class_weights)

        # 6. Metrics — accuracy + micro-averaged F1/precision/recall
        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)
            accuracy = (predictions == labels).mean()
            metrics = per_class_metric(predictions, labels)
            micro = AllClassificationMetrics.compute_micro_average(list(metrics.values()))
            return {"accuracy": round(float(accuracy), 4), **micro}

        # 7. Train
        checkpointer = _BestFinetuneCheckpointer(model, run_dir)
        trainer = _BalancedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset={"train": train_sample_hf, "val": val_hf},
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
            callbacks=[checkpointer],
            train_sampler=_make_weighted_sampler(train_hf),
        )

        logger.info("Starting training.")
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Save only the fine-tuned parts; backbone is loaded separately at inference time
        finetuned_path = run_dir / "finetuned_layers.pt"
        torch.save(
            {
                "classifier": model.classifier.state_dict(),
                "pooler": model.bert.pooler.state_dict(),
            },
            finetuned_path,
        )
        tokenizer.save_pretrained(run_dir / "tokenizer")
        (run_dir / "inference_config.json").write_text(
            json.dumps(
                {
                    "base_model": str(self.model_path),
                    "id2label": id2label,
                    "label2id": label2id,
                    "max_length": self.max_length,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Fine-tuned layers saved to %s.", finetuned_path)

        final_metrics = trainer.evaluate(test_hf)
        final_metrics = {k.removeprefix("eval_"): v for k, v in final_metrics.items()}

        (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        logger.info("Metrics saved to %s.", run_dir / "metrics.json")

        if mlflow_tracking and mlflow:
            mlflow.log_artifact(str(finetuned_path), "finetuned_layers")
            mlflow.end_run()

        return ColorTrainingResult(config=self.config, model_dir=run_dir, metrics=final_metrics)
