"""Evaluation utilities."""

import abc
from dataclasses import dataclass


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
            f"{self.feature_name}_f1_score": self.f1_score(),
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
    """Metrics for borehole metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics
    filename: str

    def get_document_level_metrics(self):
        """Get the document level metrics."""
        return {
            self.filename: {
                "elevation": self.elevation_metrics.f1_score(),
                "coordinates": self.coordinates_metrics.f1_score(),
            }
        }


@dataclass
class FileBoreholeMetadataMetrics(metaclass=abc.ABCMeta):
    """Metrics for borehole metadata."""

    borehole_metadata_metrics: list[BoreholeMetadataMetrics] = None

    def evaluate(self):
        """Evaluate the metadata metrics."""
        elevation_metrics = Metrics.from_metric_list(
            [metadata.elevation_metrics for metadata in self.borehole_metadata_metrics]
        )
        coordinates_metrics = Metrics.from_metric_list(
            [metadata.coordinates_metrics for metadata in self.borehole_metadata_metrics]
        )

        document_level_metrics = {}
        for metadata in self.borehole_metadata_metrics:
            document_level_metrics.update(metadata.get_document_level_metrics())

        return elevation_metrics, coordinates_metrics, document_level_metrics
