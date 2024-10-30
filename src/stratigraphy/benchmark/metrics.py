"""Classes for keeping track of metrics such as the F1-score, precision and recall."""

from collections import defaultdict
from collections.abc import Callable

import pandas as pd
from stratigraphy.evaluation.evaluation_dataclasses import Metrics


class OverallMetrics:
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

    def get_metrics_list(self) -> list[Metrics]:
        """Return a list of all metrics."""
        return list(self.metrics.values())


class OverallMetricsCatalog:
    """Keeps track of all different relevant metrics that are computed for a dataset."""

    def __init__(self, languages: list[str]):
        self.layer_metrics = OverallMetrics()
        self.depth_interval_metrics = OverallMetrics()
        self.groundwater_metrics = OverallMetrics()
        self.groundwater_depth_metrics = OverallMetrics()
        self.languages = languages

        # Initialize language-specific metrics
        for lang in languages:
            setattr(self, f"{lang}_layer_metrics", OverallMetrics())
            setattr(self, f"{lang}_depth_interval_metrics", OverallMetrics())

    def document_level_metrics_df(self) -> pd.DataFrame:
        """Return a DataFrame with all the document level metrics."""
        all_series = [
            self.layer_metrics.to_dataframe("F1", lambda metric: metric.f1),
            self.layer_metrics.to_dataframe("precision", lambda metric: metric.precision),
            self.layer_metrics.to_dataframe("recall", lambda metric: metric.recall),
            self.depth_interval_metrics.to_dataframe("Depth_interval_accuracy", lambda metric: metric.precision),
            self.layer_metrics.to_dataframe("Number Elements", lambda metric: metric.tp + metric.fn),
            self.layer_metrics.to_dataframe("Number wrong elements", lambda metric: metric.fp + metric.fn),
            self.groundwater_metrics.to_dataframe("groundwater", lambda metric: metric.f1),
            self.groundwater_depth_metrics.to_dataframe("groundwater_depth", lambda metric: metric.f1),
        ]
        document_level_metrics = pd.DataFrame()
        for series in all_series:
            document_level_metrics = document_level_metrics.join(series, how="outer")
        return document_level_metrics

    def metrics_dict(self) -> dict[str, float]:
        """Return a dictionary with the overall metrics."""
        # Initialize a defaultdict to automatically return 0.0 for missing keys
        result = defaultdict(lambda: None)

        # Compute the micro-average metrics for the groundwater and groundwater depth metrics
        groundwater_metrics = Metrics.micro_average(self.groundwater_metrics.metrics.values())
        groundwater_depth_metrics = Metrics.micro_average(self.groundwater_depth_metrics.metrics.values())

        # Populate the basic metrics
        result.update(
            {
                "F1": self.layer_metrics.pseudo_macro_f1() if self.layer_metrics else None,
                "recall": self.layer_metrics.macro_recall() if self.layer_metrics else None,
                "precision": self.layer_metrics.macro_precision() if self.layer_metrics else None,
                "depth_interval_accuracy": self.depth_interval_metrics.macro_precision()
                if self.depth_interval_metrics
                else None,
                "groundwater_f1": groundwater_metrics.f1,
                "groundwater_recall": groundwater_metrics.recall,
                "groundwater_precision": groundwater_metrics.precision,
                "groundwater_depth_f1": groundwater_depth_metrics.f1,
                "groundwater_depth_recall": groundwater_depth_metrics.recall,
                "groundwater_depth_precision": groundwater_depth_metrics.precision,
            }
        )

        # Add dynamic language-specific metrics only if they exist
        for lang in self.languages:
            layer_key = f"{lang}_layer_metrics"
            depth_key = f"{lang}_depth_interval_metrics"

            if getattr(self, layer_key) and getattr(self, layer_key).metrics:
                result[f"{lang}_F1"] = getattr(self, layer_key).pseudo_macro_f1()
                result[f"{lang}_recall"] = getattr(self, layer_key).macro_recall()
                result[f"{lang}_precision"] = getattr(self, layer_key).macro_precision()

            if getattr(self, depth_key) and getattr(self, depth_key).metrics:
                result[f"{lang}_depth_interval_accuracy"] = getattr(self, depth_key).macro_precision()

        return dict(result)  # Convert defaultdict back to a regular dict
