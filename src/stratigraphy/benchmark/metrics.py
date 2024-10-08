"""Classes for keeping track of metrics such as the F1-score, precision and recall."""

from collections.abc import Callable

import pandas as pd
from stratigraphy.evaluation.evaluation_dataclasses import Metrics


class DatasetMetrics:
    """Keeps track of a particular metrics for all documents in a dataset."""

    # TODO: Currently, some methods for averaging metrics are in the Metrics class.
    # (see micro_average(metric_list: list["Metrics"]). On the long run, we should refactor
    # this to have a single place where these averaging computations are implemented.

    def __init__(self):
        self.metrics: dict[str, Metrics] = {}

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
        if self.metrics and self.macro_precision() + self.macro_recall() > 0:
            return 2 * self.macro_precision() * self.macro_recall() / (self.macro_precision() + self.macro_recall())
        else:
            return 0

    def to_dataframe(self, name: str, fn: Callable[[Metrics], float]) -> pd.DataFrame:
        """Convert the metrics to a DataFrame."""
        series = pd.Series({filename: fn(metric) for filename, metric in self.metrics.items()})
        return series.to_frame(name=name)


class DatasetMetricsCatalog:
    """Keeps track of all different relevant metrics that are computed for a dataset."""

    def __init__(self):
        self.metrics: dict[str, DatasetMetrics] = {}

    def document_level_metrics_df(self) -> pd.DataFrame:
        """Return a DataFrame with all the document level metrics."""
        all_series = [
            self.metrics["layer"].to_dataframe("F1", lambda metric: metric.f1),
            self.metrics["layer"].to_dataframe("precision", lambda metric: metric.precision),
            self.metrics["layer"].to_dataframe("recall", lambda metric: metric.recall),
            self.metrics["depth_interval"].to_dataframe("Depth_interval_accuracy", lambda metric: metric.precision),
            self.metrics["layer"].to_dataframe("Number Elements", lambda metric: metric.tp + metric.fn),
            self.metrics["layer"].to_dataframe("Number wrong elements", lambda metric: metric.fp + metric.fn),
            self.metrics["groundwater"].to_dataframe("groundwater", lambda metric: metric.f1),
            self.metrics["groundwater_depth"].to_dataframe("groundwater_depth", lambda metric: metric.f1),
        ]
        document_level_metrics = pd.DataFrame()
        for series in all_series:
            document_level_metrics = document_level_metrics.join(series, how="outer")
        return document_level_metrics

    def metrics_dict(self) -> dict[str, float]:
        """Return a dictionary with the overall metrics."""
        groundwater_metrics = Metrics.micro_average(self.metrics["groundwater"].metrics.values())
        groundwater_depth_metrics = Metrics.micro_average(self.metrics["groundwater_depth"].metrics.values())

        return {
            "F1": self.metrics["layer"].pseudo_macro_f1(),
            "recall": self.metrics["layer"].macro_recall(),
            "precision": self.metrics["layer"].macro_precision(),
            "depth_interval_accuracy": self.metrics["depth_interval"].macro_precision(),
            "de_F1": self.metrics["de_layer"].pseudo_macro_f1(),
            "de_recall": self.metrics["de_layer"].macro_recall(),
            "de_precision": self.metrics["de_layer"].macro_precision(),
            "de_depth_interval_accuracy": self.metrics["de_depth_interval"].macro_precision(),
            "fr_F1": self.metrics["fr_layer"].pseudo_macro_f1(),
            "fr_recall": self.metrics["fr_layer"].macro_recall(),
            "fr_precision": self.metrics["fr_layer"].macro_precision(),
            "fr_depth_interval_accuracy": self.metrics["fr_depth_interval"].macro_precision(),
            "groundwater_f1": groundwater_metrics.f1,
            "groundwater_recall": groundwater_metrics.recall,
            "groundwater_precision": groundwater_metrics.precision,
            "groundwater_depth_f1": groundwater_depth_metrics.f1,
            "groundwater_depth_recall": groundwater_depth_metrics.recall,
            "groundwater_depth_precision": groundwater_depth_metrics.precision,
        }
