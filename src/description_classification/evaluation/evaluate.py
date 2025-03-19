"""Evaluation module."""

import logging
import os
from dataclasses import dataclass

from description_classification.utils.data_loader import LayerInformations
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.utility import count_against_ground_truth
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

classification_params = read_params("classification_params.yml")


@dataclass
class AllClassificationMetrics:
    """Metrics class."""

    global_metrics: Metrics
    language_metrics: dict[str:Metrics]

    def __repr__(self):
        cls = self.__class__.__name__
        language_metrics_repr = "\n".join(
            f"  {language}: {metrics}" for language, metrics in self.language_metrics.items()
        )
        return f"{cls}(\nglobal_metrics={self.global_metrics}\nlanguage_metrics=\n{{\n{language_metrics_repr}\n}})"

    def to_json(self) -> dict[str:float]:
        """Returns the metrics as dict.

        Returns:
            dict[str:float]: the dictionary.
        """
        return {
            **self.global_metrics.to_json("global"),
            **{
                k: v
                for language, metrics in self.language_metrics.items()
                for k, v in metrics.to_json(language).items()
            },
        }


def evaluate(layer_descriptions: list[LayerInformations]) -> AllClassificationMetrics:
    """Evaluates the predictions of the LayerInformations objects against the ground truth.

    Args:
        layer_descriptions (list[LayerDescriptionWithGroundTruth]): the LayerInformations objects

    Returns:
        AllClassificationMetrics: the holder for the metrics
    """
    global_metrics: Metrics = count_against_ground_truth(
        [layer.prediction_uscs_class for layer in layer_descriptions],
        [layer.ground_truth_uscs_class for layer in layer_descriptions],
    )

    supported_language: list[str] = classification_params["supported_language"]
    language_metrics = {
        language: count_against_ground_truth(
            [layer.prediction_uscs_class for layer in layer_descriptions if layer.language == language],
            [layer.ground_truth_uscs_class for layer in layer_descriptions if layer.language == language],
        )
        for language in supported_language
    }

    all_classification_metrics = AllClassificationMetrics(global_metrics, language_metrics)

    if mlflow_tracking:
        log_metrics_to_mlflow(all_classification_metrics)
        logger.info("Logging metrics to MLFlow")
    return all_classification_metrics


def log_metrics_to_mlflow(all_classification_metrics: AllClassificationMetrics):
    """Log metrics to MFlow."""
    for name, value in all_classification_metrics.to_json().items():
        mlflow.log_metric(name, value)
