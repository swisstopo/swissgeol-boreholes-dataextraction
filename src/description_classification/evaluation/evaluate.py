"""Evaluation module."""

import logging
import os
from dataclasses import dataclass

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

classification_params = read_params("classification_params.yml")


@dataclass
class AllClassificationMetrics:
    """Stores classification metrics at both global and language-specific levels.

    Attributes:
        global_metrics (dict[USCSClasses, Metrics]): A dictionary containing the classification metrics
            for each of the USCS classes at a global level.
        language_metrics (dict[str, dict[USCSClasses, Metrics]]): A dictionary where each key represents
            a supported language. Each value is another dictionary containing the classification metrics
            for each of the USCS classes in that language.
    """

    global_metrics: dict[USCSClasses:Metrics]
    language_metrics: dict[str : dict[USCSClasses:Metrics]]

    @staticmethod
    def macro_average(metric_list: list["Metrics"]) -> "Metrics":
        """Computes the Macro Average of a list of metrics.

        Each metric is first calculated per class and then averaged. This is useful when there is a large class
        imbalance, ensuring that all classes are given equal importance regardless of their size.

        Args:
            metric_list (list[Metrics]): The list of per-class metrics.

        Returns:
            Metrics: Macro-averaged metrics.
        """
        num_classes = len(metric_list)
        if num_classes == 0:
            return Metrics(tp=0, fp=0, fn=0)

        avg_tp = sum(metric.tp for metric in metric_list) / num_classes
        avg_fp = sum(metric.fp for metric in metric_list) / num_classes
        avg_fn = sum(metric.fn for metric in metric_list) / num_classes

        return Metrics(tp=avg_tp, fp=avg_fp, fn=avg_fn)

    @property
    def per_language_macro_avg_metrics_dict(self) -> dict[str:Metrics]:
        """Dictionary containing the metrics for each language.

        Returns:
            dict[str:Metrics]: The dictionary
        """
        return {
            k: v
            for language, metrics_dict in self.language_metrics.items()
            for k, v in self.macro_average(metrics_dict.values()).to_json(f"{language}_avg").items()
        }

    @property
    def per_class_global_metrics_dict(self) -> dict[str:Metrics]:
        """Dictionary containing the global metrics detailled for each uscs class.

        Returns:
            dict[str:Metrics]: The dictionary
        """
        return {
            k: v
            for class_, metrics in self.global_metrics.items()
            for k, v in metrics.to_json(f"global_{class_.name}").items()
        }

    @property
    def per_class_per_language_metrics_dict(self) -> dict[str:Metrics]:
        """Dictionary containing the metrics for each language, detailled for each uscs class.

        Returns:
            dict[str:Metrics]: The dictionary
        """
        return {
            k: v
            for language, metrics_dict in self.language_metrics.items()
            for class_, metrics in metrics_dict.items()
            for k, v in metrics.to_json(f"{language}_{class_.name}").items()
        }

    def to_json(self) -> dict[str:float]:
        """Returns the metrics as dict, the metrics are reduced by taking the macro average across all classes.

        Returns:
            dict[str:float]: the dictionary.
        """
        return {
            **self.macro_average(self.global_metrics.values()).to_json("global_avg"),
            **self.per_language_macro_avg_metrics_dict,
        }

    def to_json_per_class(self) -> dict[str:float]:
        """Returns the metrics as dict, detailing each classes.

        Returns:
            dict[str:float]: the dictionary.
        """
        return {
            **self.per_class_global_metrics_dict,
            **self.per_class_per_language_metrics_dict,
        }


def evaluate(layer_descriptions: list[LayerInformations], log_per_class: bool = False) -> AllClassificationMetrics:
    """Evaluates the predictions of the LayerInformations objects against the ground truth.

    Args:
        layer_descriptions (list[LayerDescriptionWithGroundTruth]): the LayerInformations objects
        log_per_class (bool, optional): whether to log per-class metrics. Defaults to False.

    Returns:
        AllClassificationMetrics: the holder for the metrics
    """
    global_metrics: dict[str, Metrics] = per_class_metrics(
        [layer.prediction_uscs_class for layer in layer_descriptions],
        [layer.ground_truth_uscs_class for layer in layer_descriptions],
    )

    supported_language: list[str] = classification_params["supported_language"]
    language_metrics = {
        language: per_class_metrics(
            [layer.prediction_uscs_class for layer in layer_descriptions if layer.language == language],
            [layer.ground_truth_uscs_class for layer in layer_descriptions if layer.language == language],
        )
        for language in supported_language
    }

    all_classification_metrics = AllClassificationMetrics(global_metrics, language_metrics)

    if mlflow_tracking:
        log_metrics_to_mlflow(all_classification_metrics, log_per_class=log_per_class)
        logger.info("Logging metrics to MLFlow")

    return all_classification_metrics


def per_class_metrics(predictions: list[USCSClasses], ground_truth: list[USCSClasses]) -> dict[USCSClasses, Metrics]:
    """Compute per-class classification metrics.

    Args:
        predictions (List[str]): List of predicted class labels.
        ground_truth (List[str]): List of actual class labels.

    Returns:
        Dict[str, Metrics]: A dictionary mapping each class to its TP, FP, FN.
    """
    all_classes = set(predictions) | set(ground_truth)
    metrics_per_class = {}

    for cls in all_classes:
        tp = sum(1 for pred, gt in zip(predictions, ground_truth, strict=False) if pred == gt == cls)
        fp = sum(1 for pred, gt in zip(predictions, ground_truth, strict=False) if pred == cls and gt != cls)
        fn = sum(1 for pred, gt in zip(predictions, ground_truth, strict=False) if pred != cls and gt == cls)

        metrics_per_class[cls] = Metrics(tp=tp, fp=fp, fn=fn)

    return metrics_per_class


def log_metrics_to_mlflow(all_classification_metrics: AllClassificationMetrics, log_per_class=False):
    """Log metrics to MLFlow with error handling."""
    # Log overall metrics
    for name, value in all_classification_metrics.to_json().items():
        mlflow.log_metric(name, value)

    # Log per-class metrics if requested
    if log_per_class:
        for name, value in all_classification_metrics.to_json_per_class().items():
            mlflow.log_metric(name, value)
