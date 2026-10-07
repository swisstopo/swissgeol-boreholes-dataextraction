"""Evaluation utilities."""

from dataclasses import dataclass

from core.benchmark_utils import Metrics


@dataclass
class BoreholeMetadataMetrics:
    """Metrics for metadata."""

    elevation_metrics: Metrics
    coordinates_metrics: Metrics
    name_metrics: Metrics
