"""Evaluation module."""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from scipy.stats import kendalltau

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask, LayerInformation
from classification.utils.file_utils import read_params
from core.benchmark_utils import Metrics
from core.mlflow_tracking import mlflow

logger = logging.getLogger(__name__)

classification_params = read_params("classification_params.yml")


@dataclass
class AllClassificationMetrics:
    """Stores classification metrics at both global and language-specific levels.

    Attributes:
        global_metrics (dict[ClassificationType.EnumMember, Metrics]): A dictionary containing the
        classification metrics for each of the classes at a global level.
        global_rank (float | None): The average Kendall's tau rank correlation between predicted and ground truth
            class orderings, computed across all layers at a global level. None for non-rank tasks.
        language_metrics (dict[str, dict[ClassificationType.EnumMember, Metrics]]): A dictionary where each key
            represents a supported language. Each value is another dictionary containing the classification metrics
            for each class in that language.
        language_ranks (dict[str, float] | None): A dictionary where each key represents a supported language and
            each value is the average Kendall's tau rank correlation between predicted and ground truth class
            orderings for that language. None for non-rank tasks.
    """

    global_metrics: dict[ClassificationSystem.EnumMember, Metrics]
    global_rank: float | None
    language_metrics: dict[str, dict[ClassificationSystem.EnumMember, Metrics]]
    language_ranks: dict[str, float] | None

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
    def per_language_rank_dict(self) -> dict[str, float]:
        """Dictionary containing rank for each language, empty for non-rank tasks.

        Returns:
            dict[str, float]: The dictionary
        """
        if self.language_ranks is None:
            return {}
        return {f"{lang}_rank": rank for lang, rank in self.language_ranks.items()}

    @property
    def per_class_global_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing the global f1, recall and precision, detailled for each class.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            f"global_{class_.name}_{k}": v
            for class_, metrics in self.global_metrics.items()
            for k, v in metrics.to_json().items()
        }

    @property
    def per_class_per_language_metrics_dict(self) -> dict[str, float]:
        """Dictionary containing the f1, recall and precision for each language, detailled for each class.

        Returns:
            dict[str, float]: The dictionary
        """
        return {
            f"{language}_{class_.name}_{k}": v
            for language, metrics_dict in self.language_metrics.items()
            for class_, metrics in metrics_dict.items()
            for k, v in metrics.to_json().items()
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
            **({"global_rank": self.global_rank} if self.global_rank is not None else {}),
            **self.global_macro_avg_dict,
            **self.global_micro_avg_dict,
            **self.per_language_rank_dict,
            **self.per_language_micro_avg_metrics_dict,
            **self.per_language_macro_avg_metrics_dict,
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


def evaluate(layer_descriptions: list[LayerInformation]) -> AllClassificationMetrics:
    """Evaluates the predictions of the LayerInformation objects against the ground truth.

    Args:
        layer_descriptions (list[LayerInformation]): the LayerInformation objects

    Returns:
        AllClassificationMetrics: the holder for the metrics
    """
    is_rank_task = (
        layer_descriptions[0].class_system.classification_task() == ClassificationTask.rank
        if layer_descriptions
        else False
    )

    global_metrics = per_class_metrics_from_layers(layer_descriptions)
    global_ranks = rank_metrics_from_layers(layer_descriptions) if is_rank_task else None

    supported_language = classification_params["supported_language"]
    language_metrics = {
        language: per_class_metrics_from_layers([layer for layer in layer_descriptions if layer.language == language])
        for language in supported_language
    }
    language_ranks = (
        {
            language: rank_metrics_from_layers([layer for layer in layer_descriptions if layer.language == language])
            for language in supported_language
        }
        if is_rank_task
        else None
    )

    all_classification_metrics = AllClassificationMetrics(
        global_metrics, global_ranks, language_metrics, language_ranks
    )

    if mlflow:
        logger.info("Logging metrics to MLFlow")
        log_metrics_to_mlflow(all_classification_metrics)

    return all_classification_metrics


def per_class_metrics_from_layers(layers: list[LayerInformation]) -> dict[ClassificationSystem.EnumMember, Metrics]:
    """Compute per-class classification metrics.

    Args:
        layers (list[LayerInformation]): the layers to compute the metrics from.

    Returns:
        Dict[ClassEnum, Metrics]: A dictionary mapping each class to its TP, FP, FN.
    """
    predictions = [layer.prediction_class for layer in layers]
    labels = [layer.ground_truth_class for layer in layers]
    return per_class_metric(predictions, labels)


def per_class_metric(
    predictions: list[list[ClassificationSystem.EnumMember]], labels: list[list[ClassificationSystem.EnumMember]]
) -> dict[ClassificationSystem.EnumMember, Metrics]:
    """Compute per-class classification metrics from the predictions and the labels.

    Supports multi-label inputs where each sample's prediction and label are lists of classes.

    Args:
        predictions (list[list[ClassificationSystem.EnumMember]]): An iterable of lists of predicted
            classes per sample.
        labels (list[list[ClassificationSystem.EnumMember]]): An iterable of lists of ground truth
            classes per sample.

    Returns:
        dict[ClassEnum, Metrics]: A dictionary mapping each class to its TP, FP, FN.
    """
    tp: defaultdict = defaultdict(int)
    fp: defaultdict = defaultdict(int)
    fn: defaultdict = defaultdict(int)

    for pred, lab in zip(predictions, labels, strict=True):
        pred_set = set(pred or [])
        lab_set = set(lab or [])
        for cls in pred_set | lab_set:
            if cls in pred_set and cls in lab_set:
                tp[cls] += 1
            elif cls in pred_set:
                fp[cls] += 1
            else:
                fn[cls] += 1

    return {cls: Metrics(tp=tp[cls], fp=fp[cls], fn=fn[cls]) for cls in tp.keys() | fp.keys() | fn.keys()}


def rank_metrics_from(
    predictions: list[list[ClassificationSystem.EnumMember]],
    labels: list[list[ClassificationSystem.EnumMember]],
) -> float:
    """Compute average Kendall's tau between parallel ranked lists of enum members.

    Args:
        predictions (list[list[ClassificationSystem.EnumMember]]): Predicted class lists in rank
            order (index 0 = highest rank) for each sample.
        labels (list[list[ClassificationSystem.EnumMember]]): Ground-truth class lists in rank
            order for each sample.

    Returns:
        float: Average Kendall's tau across all samples, or 0 if no valid pairs exist.
    """
    metrics = []
    for preds_row, labels_row in zip(predictions, labels, strict=True):
        if preds_row == labels_row:
            metrics.append(1)
            continue
        missing_to_pred = [lab for lab in labels_row if lab not in preds_row]
        missing_to_lab = [pred for pred in preds_row if pred not in labels_row]
        metric = kendalltau(preds_row + missing_to_pred, labels_row + missing_to_lab)
        statistic = metric.statistic.item()
        if not math.isnan(statistic):
            metrics.append(statistic)
    return sum(metrics) / len(metrics) if metrics else 0


def rank_metrics_from_layers(layers: list[LayerInformation]) -> float:
    """Compute the average rank correlation between predicted and ground truth class orderings.

    Args:
        layers (list[LayerInformation]): the layers to compute the rank metric from.

    Returns:
        float: the average Kendall's tau rank correlation across all layers, or 0 if there are
            no layers.
    """
    return rank_metrics_from(
        predictions=[layer.prediction_class for layer in layers],
        labels=[layer.ground_truth_class for layer in layers],
    )


def log_metrics_to_mlflow(all_classification_metrics: AllClassificationMetrics):
    """Log metrics to MLFlow with error handling."""
    # Log overall metrics
    for name, value in all_classification_metrics.to_json().items():
        mlflow.log_metric(name, value)

    # Log per-class metrics
    for name, value in all_classification_metrics.to_json_per_class().items():
        mlflow.log_metric(name, value)
