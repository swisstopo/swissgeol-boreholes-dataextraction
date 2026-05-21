"""Unified BERT training module for all classification tasks."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import datasets as hf_datasets
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import WeightedRandomSampler
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric
from classification.models.data_provider import DataSplit, TrainingDataProvider
from classification.models.model import BertModel
from core.mlflow_utils import setup_mlflow_tracking

load_dotenv()
logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"


# ---------------------------------------------------------------------------
# Loss utilities (kept here for backward compatibility; also used externally)
# ---------------------------------------------------------------------------


class WeightedLabelSmoother:
    """Label smoothing with optional per-class weighting.

    Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L539.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __init__(self, class_weights: torch.Tensor = None):
        self.class_weights = class_weights

    def __call__(
        self, model_output: SequenceClassifierOutput, labels: torch.Tensor, num_items_in_batch: torch.Tensor = None
    ) -> torch.Tensor:
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        safe_labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=safe_labels)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            nll_loss = nll_loss * weights[safe_labels.squeeze(-1)].unsqueeze(-1)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(padding_mask, 0.0)
        num_active = padding_mask.numel() - padding_mask.long().sum()
        nll = nll_loss.sum() / num_active
        smooth = smoothed_loss.sum() / (num_active * log_probs.size(-1))
        return (1 - self.epsilon) * nll + self.epsilon * smooth


def compute_trainset_weights(
    trainset: hf_datasets.Dataset, min_scale: float = 0.5, max_scale: float = 2.0
) -> torch.Tensor:
    """Compute normalised inverse-frequency class weights from a training dataset."""
    label_counts = Counter(trainset["label"])
    num_classes = max(label_counts.keys()) + 1
    raw = torch.tensor(
        [1.0 / label_counts[i] if i in label_counts else 0.0 for i in range(num_classes)], dtype=torch.float32
    )
    nonzero = raw[raw > 0]
    if len(nonzero) > 0 and nonzero.max() != nonzero.min():
        mn, mx = nonzero.min(), nonzero.max()
        scaled = min_scale + (max_scale - min_scale) * (raw - mn) / (mx - mn)
    else:
        scaled = torch.ones_like(raw)
    return scaled


# ---------------------------------------------------------------------------
# Checkpointer — saves only fine-tuned HEAD params; restores best at end
# ---------------------------------------------------------------------------


class _BestHeadCheckpointer(TrainerCallback):
    """Saves only the fine-tuned head weights on each eval; reloads best at training end.

    Tracks eval_accuracy (single eval dataset) as the selection criterion.
    Only parameters whose names start with _HEAD_PARAM_PREFIXES are saved,
    keeping checkpoint files small (~few hundred KB vs ~440 MB for full model).
    """

    def __init__(self, model: torch.nn.Module, checkpoint_dir: Path) -> None:
        self._model = model
        self._checkpoint_dir = checkpoint_dir
        self._best_metric: float = float("-inf")
        self._best_path: Path | None = None

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs
    ) -> None:
        accuracy = metrics.get("eval_accuracy", float("-inf"))
        if accuracy <= self._best_metric:
            return
        self._best_metric = accuracy
        new_path = self._checkpoint_dir / f"best_head_epoch{round(state.epoch)}.pt"
        torch.save(
            {n: p.data for n, p in self._model.named_parameters() if p.requires_grad},
            new_path,
        )
        if self._best_path and self._best_path != new_path and self._best_path.exists():
            self._best_path.unlink()
        self._best_path = new_path
        logger.info("New best val accuracy=%.4f — head saved to %s.", accuracy, new_path.name)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._best_path and self._best_path.exists():
            best = torch.load(self._best_path, weights_only=True)
            for n, p in self._model.named_parameters():
                if n in best:
                    p.data = best[n]
            logger.info("Restored best head from %s (accuracy=%.4f).", self._best_path.name, self._best_metric)


# ---------------------------------------------------------------------------
# Balanced sampler trainer
# ---------------------------------------------------------------------------


class _BalancedTrainer(Trainer):
    """Trainer that replaces the default sampler with a class-balanced one."""

    def __init__(self, *args, train_sampler: WeightedRandomSampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, dataset=None) -> WeightedRandomSampler:
        return self._train_sampler


def _make_weighted_sampler(dataset: hf_datasets.Dataset) -> WeightedRandomSampler:
    labels = dataset["label"]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.float)
    return WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Outcome of a single BertTrainer.run() call."""

    name: str
    model_dir: Path
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Unified trainer
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BertTrainer:
    """Trains a BERT model for any classification task.

    Accepts any TrainingDataProvider, making it task-agnostic.
    Always saves only fine-tuned head layers (classifier + pooler + optional encoder
    layers), never the full backbone (~440 MB).

    Args:
        provider:              Data provider supplying labels and train/val/test splits.
        model_path:            Local BERT checkpoint or HuggingFace model ID.
        out_directory:         Root directory; a timestamped sub-directory is created per run.
        unfreeze_layers:       Layers to unfreeze (e.g. ["classifier", "pooler", "layer_11"]).
        batch_size:            Per-device batch size.
        num_epochs:            Training epochs.
        learning_rate:         Peak learning rate.
        weight_decay:          L2 regularisation.
        warmup_ratio:          Fraction of steps for LR warm-up.
        lr_scheduler_type:     LR schedule (e.g. "cosine", "cosine_with_restarts").
        max_grad_norm:         Gradient clipping threshold.
        max_length:            Maximum token sequence length.
        use_class_balancing:   Weight loss by inverse class frequency.
        use_balanced_sampler:  Up-sample rare classes in training batches.
    """

    provider: TrainingDataProvider
    model_path: str | Path
    out_directory: Path
    unfreeze_layers: list[str] = field(default_factory=lambda: ["classifier", "pooler"])
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 5.0
    max_length: int = 128
    use_class_balancing: bool = False
    use_balanced_sampler: bool = True

    def run(self) -> TrainingResult:
        """Train, select best checkpoint on val, evaluate on test, and save head weights."""
        run_dir = self.out_directory / self.provider.name / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        if mlflow_tracking:
            setup_mlflow_tracking(
                run_id=None,
                experiment_name="BERT classification",
                runname=self.provider.name,
                params={
                    "model_path": str(self.model_path),
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "unfreeze_layers": str(self.unfreeze_layers),
                    "use_class_balancing": self.use_class_balancing,
                    "use_balanced_sampler": self.use_balanced_sampler,
                },
            )

        labels = self.provider.get_labels()
        bert = BertModel.from_labels(self.model_path, labels)
        bert.freeze_all_layers()
        bert.unfreeze_list(self.unfreeze_layers)
        bert.model.train()
        logger.info(
            "Trainable params: %d",
            sum(p.numel() for p in bert.model.parameters() if p.requires_grad),
        )

        logger.info("Loading and tokenising splits for '%s'.", self.provider.name)
        split: DataSplit = self.provider.get_split(bert.tokenizer, self.max_length)
        logger.info(
            "Train: %d | Val: %d | Test: %d samples.",
            len(split.train),
            len(split.val),
            len(split.test),
        )

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

        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            logits, gold = eval_pred
            preds = logits.argmax(axis=-1)
            accuracy = float((preds == gold).mean())
            micro = AllClassificationMetrics.compute_micro_average(list(per_class_metric(preds, gold).values()))
            return {"accuracy": round(accuracy, 4), **micro}

        compute_loss_func = None
        if self.use_class_balancing:
            compute_loss_func = WeightedLabelSmoother(class_weights=compute_trainset_weights(split.train))

        checkpointer = _BestHeadCheckpointer(bert.model, run_dir)
        trainer_cls = _BalancedTrainer if self.use_balanced_sampler else Trainer
        trainer_kwargs: dict = dict(
            model=bert.model,
            args=training_args,
            train_dataset=split.train,
            eval_dataset=split.val,
            processing_class=bert.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=bert.tokenizer),
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
            callbacks=[checkpointer],
        )
        if self.use_balanced_sampler:
            trainer_kwargs["train_sampler"] = _make_weighted_sampler(split.train)

        trainer = trainer_cls(**trainer_kwargs)

        logger.info("Starting training for '%s'.", self.provider.name)
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Save fine-tuned head (small file — backbone loaded separately at inference)
        head_path = run_dir / "finetuned_layers.pt"
        torch.save(
            {n: p.data for n, p in bert.model.named_parameters() if p.requires_grad},
            head_path,
        )
        bert.tokenizer.save_pretrained(run_dir / "tokenizer")
        (run_dir / "inference_config.json").write_text(
            json.dumps(
                {"base_model": str(self.model_path), "id2label": bert.id2label, "max_length": self.max_length},
                indent=2,
            ),
            encoding="utf-8",
        )

        test_metrics = trainer.evaluate(split.test)
        test_metrics = {k.removeprefix("eval_"): v for k, v in test_metrics.items()}
        for extra_name, extra_test in split.extra_tests.items():
            extra = trainer.evaluate(extra_test)
            test_metrics.update({f"{extra_name}_{k.removeprefix('eval_')}": v for k, v in extra.items()})
        (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

        logger.info(
            "Run complete. Head saved to %s. Test accuracy: %.4f",
            head_path,
            test_metrics.get("accuracy", float("nan")),
        )

        if mlflow_tracking:
            from core.mlflow_tracking import mlflow as _mlflow

            _mlflow.end_run()

        return TrainingResult(name=self.provider.name, model_dir=run_dir, metrics=test_metrics)
