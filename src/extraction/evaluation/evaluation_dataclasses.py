"""Evaluation utilities."""

from dataclasses import dataclass

from core.benchmark_utils import Metrics


@dataclass
class BoreholeMetadataMetrics:
    """Metrics for metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics
    name_metrics: Metrics

    def to_json(self) -> dict[str, dict]:
        """Converts the object to a dictionary.

        Returns:
            dict[str, dict]: The object as a dictionary.
        """
        return {
            "elevation": self.elevation_metrics.to_json(),
            "coordinates": self.coordinates_metrics.to_json(),
            "name": self.name_metrics.to_json(),
        }

    @classmethod
    def from_json(cls, json: dict) -> "BoreholeMetadataMetrics":
        """Construct a BoreholeMetadataMetrics instance from a dictionary produced by `to_json`.

        Args:
            json (dict): Dictionary with elevation, coordinates, and name metrics.

        Returns:
            BoreholeMetadataMetrics: The reconstructed metadata metrics object.
        """
        return BoreholeMetadataMetrics(
            elevation_metrics=Metrics.from_json(json["elevation"]),
            coordinates_metrics=Metrics.from_json(json["coordinates"]),
            name_metrics=Metrics.from_json(json["name"]),
        )
