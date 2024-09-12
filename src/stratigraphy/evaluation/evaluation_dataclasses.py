"""Evaluation utilities."""

import abc
from dataclasses import dataclass

import pandas as pd


@dataclass
class Metrics(metaclass=abc.ABCMeta):
    """Metrics for metadata."""

    tp: int
    fp: int
    fn: int
    feature_name: str

    def precision(self) -> float:
        """Calculates the precision.

        Returns:
            float: The precision.
        """
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0

    def recall(self) -> float:
        """Calculates the recall.

        Returns:
            float: The recall.
        """
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0

    def f1_score(self) -> float:
        """Calculates the F1 score.

        Returns:
            float: The F1 score.
        """
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            f"{self.feature_name}_tp": self.tp,
            f"{self.feature_name}_fp": self.fp,
            f"{self.feature_name}_fn": self.fn,
            f"{self.feature_name}_precision": self.precision(),
            f"{self.feature_name}_recall": self.recall(),
            f"{self.feature_name}_f1": self.f1_score(),
        }

    @staticmethod
    def from_metric_list(metric_list: list["Metrics"]) -> "Metrics":
        """Converts a list of metrics to a metric.

        Args:
            metric_list (list): The list of metrics.

        Returns:
            Metrics: Combined metrics.
        """
        tp = sum([metric.tp for metric in metric_list])
        fp = sum([metric.fp for metric in metric_list])
        fn = sum([metric.fn for metric in metric_list])

        # assert that the feature name is the same for all metrics
        assert all([metric.feature_name == metric_list[0].feature_name for metric in metric_list])

        return Metrics(tp=tp, fp=fp, fn=fn, feature_name=metric_list[0].feature_name)


@dataclass
class BoreholeMetadataMetrics(metaclass=abc.ABCMeta):
    """Metrics for metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            **self.elevation_metrics.to_json(),
            **self.coordinates_metrics.to_json(),
        }


@dataclass
class FileBoreholeMetadataMetrics(BoreholeMetadataMetrics):
    """Single file Metrics for borehole metadata."""

    filename: str

    def get_document_level_metrics(self) -> pd.DataFrame:
        """Get the document level metrics."""
        return pd.DataFrame(
            data={
                "document_name": [self.filename],
                "elevation": [self.elevation_metrics.f1_score()],
                "coordinates": [self.coordinates_metrics.f1_score()],
            }
        )


@dataclass
class OverallBoreholeMetadataMetrics(metaclass=abc.ABCMeta):
    """Metrics for borehole metadata."""

    borehole_metadata_metrics: list[BoreholeMetadataMetrics] = None

    def __init__(self):
        """Initializes the FileBoreholeMetadataMetrics object."""
        self.borehole_metadata_metrics = []

    def get_cumulated_metrics(self) -> dict:
        """Evaluate the metadata metrics."""
        elevation_metrics = Metrics.from_metric_list(
            [metadata.elevation_metrics for metadata in self.borehole_metadata_metrics]
        )
        coordinates_metrics = Metrics.from_metric_list(
            [metadata.coordinates_metrics for metadata in self.borehole_metadata_metrics]
        )
        return BoreholeMetadataMetrics(
            elevation_metrics=elevation_metrics, coordinates_metrics=coordinates_metrics
        ).to_json()

    def get_document_level_metrics(self) -> pd.DataFrame:
        """Get the document level metrics."""
        document_level_metrics = pd.DataFrame(columns=["document_name", "elevation", "coordinates"])
        for metadata in self.borehole_metadata_metrics:
            document_level_metrics = pd.concat([document_level_metrics, metadata.get_document_level_metrics()])

        return document_level_metrics
