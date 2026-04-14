"""Evaluation utilities."""

import abc
from dataclasses import dataclass, field

import pandas as pd

from core.benchmark_utils import Metrics


@dataclass
class BoreholeMetadataMetrics(metaclass=abc.ABCMeta):
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
            **self.elevation_metrics.to_json("elevation"),
            **self.coordinates_metrics.to_json("coordinate"),
            **self.name_metrics.to_json("borehole_name"),
        }


@dataclass
class FileBoreholeMetadataMetrics(BoreholeMetadataMetrics):
    """Single file Metrics for borehole metadata."""

    filename: str

    def get_document_level_metrics(self) -> pd.DataFrame:
        """Get the document level metrics."""
        return pd.DataFrame(
            data={
                "elevation": [self.elevation_metrics.f1],
                "coordinate": [self.coordinates_metrics.f1],
                "borehole_name": [self.name_metrics.f1],
            },
            index=[self.filename],
        )


@dataclass
class OverallBoreholeMetadataMetrics:
    """Metrics for borehole metadata."""

    borehole_metadata_metrics: list[BoreholeMetadataMetrics] = field(default_factory=list)

    def add_metadata_metrics(self, borehole_metadata_metrics: BoreholeMetadataMetrics):
        """Append per-file metadata metrics to the collection.

        Args:
            borehole_metadata_metrics (BoreholeMetadataMetrics): The metrics for a single file to add.
        """
        self.borehole_metadata_metrics.append(borehole_metadata_metrics)

    def get_cumulated_metrics(self) -> BoreholeMetadataMetrics:
        """Compute metadata metrics across all collected files.

        Returns:
            BoreholeMetadataMetrics: Micro-average of elevation, coordinate, and name metrics
                across all files in the collection.
        """
        elevation_metrics = Metrics.micro_average(
            [metadata.elevation_metrics for metadata in self.borehole_metadata_metrics]
        )
        coordinates_metrics = Metrics.micro_average(
            [metadata.coordinates_metrics for metadata in self.borehole_metadata_metrics]
        )
        name_metrics = Metrics.micro_average([metadata.name_metrics for metadata in self.borehole_metadata_metrics])
        return BoreholeMetadataMetrics(
            elevation_metrics=elevation_metrics, coordinates_metrics=coordinates_metrics, name_metrics=name_metrics
        )

    def get_document_level_metrics(self) -> pd.DataFrame:
        """Get metrics aggregated at the document level.

        Returns:
            pd.DataFrame: A DataFrame indexed by document names with columns:
                - elevation: F1 score for elevation predictions
                - coordinate: F1 score for coordinate predictions
                - borehole_name: F1 score for borehole name predictions

        Example:
                         elevation  coordinate  borehole_name
            doc1.pdf     1.00       1.00        0.00
            doc2.pdf     1.00       0.00        1.00
        """
        # Get a dataframe per document, concatenate, and sort by index (document name)
        return pd.concat(
            [metadata.get_document_level_metrics() for metadata in self.borehole_metadata_metrics]
        ).sort_index()
