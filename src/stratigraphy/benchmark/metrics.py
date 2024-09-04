"""Classes for keeping track of metrics such as the F1-score, precision and recall."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass
class Metrics:
    """Computes F-score metrics.

    See also https://en.wikipedia.org/wiki/F-score

    Args:
        tp (int): The true positive count
        fp (int): The false positive count
        fn (int): The false negative count
    """

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """Calculate the precision."""
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return 0

    @property
    def recall(self) -> float:
        """Calculate the recall."""
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return 0

    @property
    def f1(self) -> float:
        """Calculate the F1 score."""
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0


class DatasetMetrics:
    """Keeps track of a particular metrics for all documents in a dataset."""

    def __init__(self):
        self.metrics: dict[str, Metrics] = {}

    def overall_metrics(self) -> Metrics:
        """Can be used to compute micro averages."""
        return Metrics(
            tp=sum(metric.tp for metric in self.metrics.values()),
            fp=sum(metric.fp for metric in self.metrics.values()),
            fn=sum(metric.fn for metric in self.metrics.values()),
        )

    def macro_f1(self) -> float:
        """Compute the macro F1 score."""
        if self.metrics:
            return sum([metric.f1 for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def macro_precision(self) -> float:
        """Compute the macro precision score."""
        if self.metrics:
            return sum([metric.precision for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def macro_recall(self) -> float:
        """Compute the macro recall score."""
        if self.metrics:
            return sum([metric.recall for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def pseudo_macro_f1(self) -> float:
        """Compute a "pseudo" macro F1 score by using the values of the macro precision and macro recall.

        TODO: we probably should not use this metric, and use the proper macro F1 score instead.
        """
        if self.metrics:
            return 2 * self.macro_precision() * self.macro_recall() / (self.macro_precision() + self.macro_recall())
        else:
            return 0

    def to_dataframe(self, name: str, fn: Callable[[Metrics], float]) -> pd.DataFrame:
        series = pd.Series({filename: fn(metric) for filename, metric in self.metrics.items()})
        return series.to_frame(name=name)
