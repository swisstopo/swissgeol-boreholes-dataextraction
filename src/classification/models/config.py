"""TODO."""

from pydantic import BaseModel


class ExperimentHyperparameters(BaseModel):
    """TODO."""

    batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    max_grad_norm: float
    num_epochs: int
    weight_decay: float
    warmup_ratio: float


class ExperimentDatasetConfig(BaseModel):
    """TODO."""

    ground_truth: str
    classification_system: str


class ExperimentConfig(BaseModel):
    """TODO."""

    classification_system: str
    experiment_name: str
    hyperparameters: ExperimentHyperparameters
    model_path: str | None = None
    training_sets: list[ExperimentDatasetConfig]
    test_sets: list[ExperimentDatasetConfig]
    unfreeze_layers: list[str] = []
    use_class_balancing: bool
