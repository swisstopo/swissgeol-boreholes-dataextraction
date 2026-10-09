"""Evaluation utilities."""

from dataclasses import dataclass

from pydantic import ConfigDict, Field

from core.benchmark_utils import Metrics


@dataclass
class BoreholeMetadataMetrics:
    """Metrics for metadata."""

    model_config = ConfigDict(populate_by_name=True)

    elevation_metrics: Metrics = Field(alias="elevation")
    coordinates_metrics: Metrics = Field(alias="coordinates")
    name_metrics: Metrics = Field(alias="name")
