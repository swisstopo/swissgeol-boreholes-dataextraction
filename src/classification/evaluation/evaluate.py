"""Evaluation module."""

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from classification.utils.classification_classes import ClassEnum
from classification.utils.data_loader import LayerInformations
from extraction.evaluation.evaluation_dataclasses import Metrics
from utils.file_utils import read_params

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

classification_params = read_params("classification_params.yml")


@dataclass
class AllClassificationMetrics:
    """Stores classification metrics at both global and language-specific levels.

    Attributes:
        global_metrics (dict[ClassEnum, Metrics]): A dictionary containing the classification metrics
            for each of the classes at a global level.
        language_metrics (dict[str, dict[ClassEnum, Metrics]]): A dictionary where each key represents
            a supported language. Each value is another dictionary containing the classification metrics
            for each of the classes in that language.
    """

    global_metrics: dict[ClassEnum, Metrics]
    language_metrics: dict[str, dict[ClassEnum, Metrics]]

    @staticmethod
    def compute_macro_average(metric_list: list[Metrics]) -> dict[str, float]:
        """Computes the Macro Average of a list of metrics.

        Each metric is first calculated per class and then averaged. This is useful when there is a large class
        imbalance, ensuring that all classes are given equal importance regardless of their size.

        Args:
            metric_list (list[Metrics]): The list of per-class metrics.

        Returns:
            dict[str, float]: The dict with macro-averaged metrics, with keys macro_precision, macro_recall and
                macro_f1.
        """
        # Filter out classes with no presence in the data (i.e. tp + fn + fp == 0) to avoid penalizing macro-averaging
        # by including classes that do not appear in the data. This follows sklearn's approach and is likely best
        # practice for computing macro-averaged metrics.
        valid_metrics = [m for m in metric_list if (m.tp + m.fn + m.fp) > 0]

        if not valid_metrics:
            return {"macro_precision": 0, "macro_recall": 0, "macro_f1": 0}

        precisions = [metrics.precision for metrics in valid_metrics]
        recalls = [metrics.recall for metrics in valid_metrics]
        f1s = [metrics.f1 for metrics in valid_metrics]

        return {
            "macro_precision": round(sum(precisions) / len(precisions), 4),
            "macro_recall": round(sum(recalls) / len(recalls), 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4),
        }

    @staticmethod
    def compute_micro_average(metric_list: list[Metrics]) -> dict[str, float]:
        """Computes the Micro Average of a list of metrics.

        Unlike macro averaging, micro averaging aggregates true positives, false positives, and false negatives across
        all classes before computing precision, recall, and F1-score.
        This gives more weight to larger classes, making it suitable when class imbalance is present.

        Args:
            metric_list (list[Metrics]): The list of per-class metrics.

        Returns:
            dict[str, float]: The dict with micro-averaged metrics, with keys micro_precision, micro_recall,
                and micro_f1.
        """
        if not metric_list:
            return {"micro_precision": 0, "micro_recall": 0, "micro_f1": 0}

        all_aggregated_metric = Metrics.micro_average(metric_list)

        return {
            "micro_precision": round(all_aggregated_metric.precision, 4),
            "micro_recall": round(all_aggregated_metric.recall, 4),
            "micro_f1": round(all_aggregated_metric.f1, 4),
        }

    @property
    def global_macro_avg_dict(self) -> dict[str, float]:
        """Dictionary containing the f1, recall and precision, macro averaged across all classes.

        Returns:
            dict[str, float]: The dictionary
        """
        return {f"global_{k}": v for k, v in self.compute_macro_average(self.global_metrics.values()).items()}

    @property
    def global_micro_avg_dict(self) -> dict[str, float]:
        """Dictionary containing the f1, recall and precision, micro averaged across all classes.

        Returns:
            dict[str, float]: The dictionary
        """
        return {f"global_{k}": v for k, v in self.compute_micro_average(self.global_metrics.values()).items()}

    @property
    def per_language_macro_avg_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing f1, recall and precision for each language, macro averaged across all classes.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            f"{language}_{k}": v
            for language, metrics_dict in self.language_metrics.items()
            for k, v in self.compute_macro_average(metrics_dict.values()).items()
        }

    @property
    def per_language_micro_avg_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing f1, recall and precision for each language, micro averaged across all classes.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            f"{language}_{k}": v
            for language, metrics_dict in self.language_metrics.items()
            for k, v in self.compute_micro_average(metrics_dict.values()).items()
        }

    @property
    def per_class_global_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing the global f1, recall and precision, detailled for each class.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            k: v
            for class_, metrics in self.global_metrics.items()
            for k, v in metrics.to_json(f"global_{class_.name}").items()
        }

    @property
    def per_class_per_language_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing the f1, recall and precision for each language, detailled for each class.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            k: v
            for language, metrics_dict in self.language_metrics.items()
            for class_, metrics in metrics_dict.items()
            for k, v in metrics.to_json(f"{language}_{class_.name}").items()
        }

    @property
    def per_class_all_metrics_dict(self) -> dict[str, float]:
        return {**self.per_class_global_metrics_dict, **self.per_class_per_language_metrics_dict}

    def to_json(self) -> dict[str, float]:
        """Returns the metrics as dict, the metrics are reduced by taking the macro average across all classes.

        Returns:
            dict[str, float]: the dictionary.
        """
        return {
            **self.global_macro_avg_dict,
            **self.per_language_macro_avg_metrics_dict,
            **self.global_micro_avg_dict,
            **self.per_language_micro_avg_metrics_dict,
        }

    def to_json_per_class(self) -> dict[str, float]:
        """Returns the metrics as dict, detailing each classes.

        Returns:
            dict[str, float]: the dictionary.
        """
        return {
            **self.per_class_global_metrics_dict,
            **self.per_class_per_language_metrics_dict,
        }


def evaluate(layer_descriptions: list[LayerInformations]) -> AllClassificationMetrics:
    """Evaluates the predictions of the LayerInformations objects against the ground truth.

    Args:
        layer_descriptions (list[LayerInformations]): the LayerInformations objects

    Returns:
        AllClassificationMetrics: the holder for the metrics
    """
    global_metrics: dict[ClassEnum, Metrics] = per_class_metrics_from_layers(layer_descriptions)

    supported_language: list[str] = classification_params["supported_language"]
    language_metrics: dict[str, dict[ClassEnum, Metrics]] = {
        language: per_class_metrics_from_layers([layer for layer in layer_descriptions if layer.language == language])
        for language in supported_language
    }

    all_classification_metrics = AllClassificationMetrics(global_metrics, language_metrics)

    if mlflow_tracking:
        log_metrics_to_mlflow(all_classification_metrics)
        logger.info("Logging metrics to MLFlow")

    return all_classification_metrics


def per_class_metrics_from_layers(layers: list[LayerInformations]) -> dict[ClassEnum, Metrics]:
    """Compute per-class classification metrics.

    Args:
        layers (list[LayerInformations]): the layers to compute the metrics from.

    Returns:
        Dict[ClassEnum, Metrics]: A dictionary mapping each class to its TP, FP, FN.
    """
    predictions = [layer.prediction_class for layer in layers]
    labels = [layer.ground_truth_class for layer in layers]
    return per_class_metric(predictions, labels)


def per_class_metric(predictions: Iterable, labels: Iterable) -> dict[ClassEnum, Metrics]:
    """Compute per-class classification metrics from the predictions and the labels.

    Args:
        predictions (Iterable): An iterable containing the prediction for each sample.
        labels (Iterable): An iterable containing the ground truth label for each sample.

    Returns:
        dict[ClassEnum, Metrics]: A dictionary mapping each class to its TP, FP, FN.
    """
    metrics_per_class = {}
    classes: set[ClassEnum] = set(predictions) | set(labels)
    for cls in classes:
        tp = sum(1 for pred, lab in zip(predictions, labels, strict=True) if pred == lab == cls)
        fp = sum(1 for pred, lab in zip(predictions, labels, strict=True) if pred == cls and lab != cls)
        fn = sum(1 for pred, lab in zip(predictions, labels, strict=True) if pred != cls and lab == cls)

        metrics_per_class[cls] = Metrics(tp=tp, fp=fp, fn=fn)

    return metrics_per_class


def log_metrics_to_mlflow(all_classification_metrics: AllClassificationMetrics):
    """Log metrics to MLFlow with error handling."""
    # Log overall metrics
    for name, value in all_classification_metrics.to_json().items():
        mlflow.log_metric(name, value)

    # Log per-class metrics
    for name, value in all_classification_metrics.to_json_per_class().items():
        mlflow.log_metric(name, value)
