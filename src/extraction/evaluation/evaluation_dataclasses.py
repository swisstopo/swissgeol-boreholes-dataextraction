"""Evaluation utilities."""

from dataclasses import dataclass, field

from core.benchmark_utils import Metrics


@dataclass
class BoreholeMetadataMetrics:
    """Metrics for metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics
    name_metrics: Metrics

    def to_json(self) -> dict[str, float]:
        """Converts the object to a dictionary.

        Returns:
            dict[str, float]: The object as a dictionary.
        """
        return {
            "elevation": self.elevation_metrics.to_json(),
            "coordinates": self.coordinates_metrics.to_json(),
            "name": self.name_metrics.to_json(),
        }

    @classmethod
    def from_json(cls, json: dict) -> Metrics:
        """TODO."""
        return BoreholeMetadataMetrics(
            elevation_metrics=Metrics.from_json(json["elevation"]),
            coordinates_metrics=Metrics.from_json(json["coordinates"]),
            name_metrics=Metrics.from_json(json["name"]),
        )


@dataclass
class OverallBoreholeMetadataMetrics:
    """Metrics for borehole metadata."""

    borehole_metadata_metrics: list[BoreholeMetadataMetrics] = field(default_factory=list)

    def add_metadata_metrics(self, borehole_metadata_metrics: BoreholeMetadataMetrics) -> None:
        """Append per-file metadata metrics to the collection.

        Args:
            borehole_metadata_metrics (BoreholeMetadataMetrics): The metrics for a single file to add.
        """
        self.borehole_metadata_metrics.append(borehole_metadata_metrics)
