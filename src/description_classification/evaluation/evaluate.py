"""Evaluate module."""

from dataclasses import dataclass

from description_classification.classifiers.classifiers import LayerUSCSPrediction, USCSClasses
from description_classification.utils.data_loader import LayerUSCSGroundTruth
from sklearn.metrics import f1_score, precision_score, recall_score
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")


@dataclass
class ClassificationMetrics:
    """Metric."""

    f1: float
    precision: float
    recall: float

    @classmethod
    def evaluate(cls, pred_classes: list[USCSClasses], true_classes: list[USCSClasses]):
        """_summary_.

        Args:
            pred_classes (list[USCSClasses]): _description_
            true_classes (list[USCSClasses]): _description_

        Returns:
            _type_: _description_
        """
        f1 = f1_score(true_classes, pred_classes, average="weighted", zero_division=0)
        precision = precision_score(true_classes, pred_classes, average="weighted", zero_division=0)
        recall = recall_score(true_classes, pred_classes, average="weighted", zero_division=0)

        return cls(f1, precision, recall)


@dataclass
class AllClassificationMetrics:
    """Metric."""

    global_metrics: ClassificationMetrics
    language_metrics: dict[str:ClassificationMetrics]

    def __repr__(self):
        cls = self.__class__.__name__
        language_metrics_repr = "\n".join(
            f"  {language}: {metrics}" for language, metrics in self.language_metrics.items()
        )
        return f"{cls}(\nglobal_metrics={self.global_metrics}\nlanguage_metrics=\n{{\n{language_metrics_repr}\n}})"


def evaluate(predictions: list[LayerUSCSPrediction], ground_truth: list[LayerUSCSGroundTruth]):
    """_summary_.

    Args:
        predictions (list[LayerUSCSPrediction]): _description_
        ground_truth (list[LayerUSCSGroundTruth]): _description_

    Returns:
        _type_: _description_
    """
    # call evaluate for all lang and gloabal
    supported_language = classification_params["supported_language"]
    global_metrics = ClassificationMetrics.evaluate(
        [pred.uscs_class.value for pred in predictions], [gt.uscs_class.value for gt in ground_truth]
    )
    language_metrics = {
        language: ClassificationMetrics.evaluate(
            [pred.uscs_class.value for pred in predictions if pred.language == language],
            [gt.uscs_class.value for gt in ground_truth if gt.language == language],
        )
        for language in supported_language
    }

    return AllClassificationMetrics(global_metrics, language_metrics)
