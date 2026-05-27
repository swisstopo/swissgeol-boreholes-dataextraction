"""Pydantic configuration models for BERT classification experiments."""

from pydantic import BaseModel


class ExperimentHyperparameters(BaseModel):
    """Training hyperparameters shared across all datasets in an experiment."""

    batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    max_grad_norm: float
    num_epochs: int
    weight_decay: float
    warmup_ratio: float


class ExperimentDatasetConfig(BaseModel):
    """Configuration for a single ground-truth dataset used during training or evaluation."""

    ground_truth: str
    classification_system: str


class ExperimentConfig(BaseModel):
    """Top-level configuration for a BERT fine-tuning experiment."""

    classification_system: str
    experiment_name: str
    hyperparameters: ExperimentHyperparameters
    model_path: str | None = None
    training_sets: list[ExperimentDatasetConfig]
    test_sets: list[ExperimentDatasetConfig]
    unfreeze_layers: list[str] = []
    use_class_balancing: bool
