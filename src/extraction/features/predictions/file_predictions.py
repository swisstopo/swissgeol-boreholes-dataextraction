"""Classes for predictions per PDF file."""

import dataclasses

from core.benchmark_utils import Metrics
from extraction.evaluation.evaluation_dataclasses import BoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterMetrics
from extraction.features.metadata.metadata import FileMetadata
from extraction.features.predictions.borehole_predictions import BoreholePredictions


@dataclasses.dataclass
class FilePredictions:
    """A class to represent predictions for a single file."""

    borehole_predictions_list: list[BoreholePredictions]
    file_metadata: FileMetadata
    file_name: str

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            **self.file_metadata.to_json(),
            "boreholes": [borehole.to_json() for borehole in self.borehole_predictions_list],
        }


@dataclasses.dataclass
class FilePredictionsMetrics:
    """Evaluation metrics for a single extracted file, covering all extraction categories."""

    language: str
    layer_metrics: Metrics
    depth_interval_metrics: Metrics
    material_description_metrics: Metrics
    gw_metrics: GroundwaterMetrics
    metadata_metrics: BoreholeMetadataMetrics

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "language": self.language,
            "layer_metrics": self.layer_metrics.to_json(),
            "depth_interval_metrics": self.depth_interval_metrics.to_json(),
            "material_description_metrics": self.material_description_metrics.to_json(),
            "gw_metrics": self.gw_metrics.to_json(),
            "metadata_metrics": self.metadata_metrics.to_json(),
        }

    @classmethod
    def from_json(cls, json: dict, filename: str) -> "FilePredictionsMetrics":
        """Construct a FilePredictionsMetrics instance from a dictionary produced by `to_json`.

        Args:
            json (dict): Dictionary with layer_metrics, depth_interval_metrics, material_description_metrics,
                gw_metrics, and metadata_metrics.
            filename (str): Linked filename.

        Returns:
            FilePredictionsMetrics: The reconstructed metrics object.
        """
        return FilePredictionsMetrics(
            language=json["language"],
            layer_metrics=Metrics.from_json(json["layer_metrics"]),
            depth_interval_metrics=Metrics.from_json(json["depth_interval_metrics"]),
            material_description_metrics=Metrics.from_json(json["material_description_metrics"]),
            gw_metrics=GroundwaterMetrics.from_json(json["gw_metrics"], filename),
            metadata_metrics=BoreholeMetadataMetrics.from_json(json["metadata_metrics"]),
        )


@dataclasses.dataclass
class FilePredictionsWithMetrics:
    """A class to represent predictions for a single file."""

    filename: str
    file_metadata: FileMetadata
    boreholes: list[BoreholePredictions]
    metrics: FilePredictionsMetrics | None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "file_metadata": self.file_metadata.to_json(),
            "boreholes": [borehole.to_json() for borehole in self.boreholes],
            "metrics": self.metrics.to_json() if self.metrics else None,
        }

    @classmethod
    def from_json(cls, json: dict, filename: str) -> "FilePredictionsWithMetrics":
        """Construct a FilePredictionsWithMetrics instance from a dictionary produced by `to_json`.

        Args:
            json (dict): Dictionary with filename, boreholes, and metrics.
            filename (str): Filename of the document.

        Returns:
            FilePredictionsWithMetrics: The reconstructed predictions object.
        """
        return FilePredictionsWithMetrics(
            filename=filename,
            file_metadata=FileMetadata.from_json(json["file_metadata"], filename),
            boreholes=[BoreholePredictions.from_json(bh_data, filename) for bh_data in json["boreholes"]],
            metrics=FilePredictionsMetrics.from_json(json["metrics"], filename) if json.get("metrics") else None,
        )
