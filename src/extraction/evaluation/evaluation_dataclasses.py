"""Evaluation utilities."""

from pydantic import BaseModel, ConfigDict, Field

from core.benchmark_utils import Metrics


class BoreholeMetadataMetrics(BaseModel):
    """Metrics for metadata."""

    model_config = ConfigDict(populate_by_name=True)

    elevation_metrics: Metrics = Field(alias="elevation")
    coordinates_metrics: Metrics = Field(alias="coordinates")
    name_metrics: Metrics = Field(alias="name")
