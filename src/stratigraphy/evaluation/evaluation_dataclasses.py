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


@dataclass
class BoreholeMetadataMetrics(metaclass=abc.ABCMeta):
    """Metrics for borehole metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics


@dataclass
class FileBoreholeMetadataMetrics(BoreholeMetadataMetrics):
    """Metrics for borehole metadata."""

    filename: str
