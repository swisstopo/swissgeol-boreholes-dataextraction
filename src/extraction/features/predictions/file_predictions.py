"""Classes for predictions per PDF file."""

from pydantic import BaseModel

from core.benchmark_utils import Metrics
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterMetrics
from extraction.features.metadata.metadata import FileMetadata
from extraction.features.predictions.borehole_predictions import BoreholePredictions


class FilePredictions(BaseModel):
    """A class to represent predictions for a single file."""

    borehole_predictions_list: list[BoreholePredictions]
    file_metadata: FileMetadata
    file_name: str


class FilePredictionsMetrics(BaseModel):
    """Evaluation metrics for a single extracted file, covering all extraction categories."""

    language: str
    layer_metrics: Metrics
    depth_interval_metrics: Metrics
    material_description_metrics: Metrics
    gw_metrics: GroundwaterMetrics
    metadata_metrics: BoreholeMetadataMetrics


class FilePredictionsWithMetrics(BaseModel):
    """Predictions for a single PDF file, including optional per-category evaluation metrics."""

    filename: str
    file_metadata: FileMetadata
    boreholes: list[BoreholePredictions]
    metrics: FilePredictionsMetrics | None
