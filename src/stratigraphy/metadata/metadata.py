"""Metadata for stratigraphy data."""

import abc
import math
from dataclasses import dataclass
from typing import NamedTuple

import fitz
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.metadata.coordinates.coordinate_extraction import Coordinate, CoordinateExtractor
from stratigraphy.metadata.elevation.elevation_extraction import ElevationExtractor, ElevationInformation
from stratigraphy.util.language_detection import detect_language_of_document
from stratigraphy.util.util import read_params


class PageDimensions(NamedTuple):
    """Class for page dimensions."""

    width: float
    height: float

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {"width": self.width, "height": self.height}


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


@dataclass
class BoreholeMetadata(metaclass=abc.ABCMeta):
    """Metadata for stratigraphy data."""

    elevation: ElevationInformation | None = None
    coordinates: Coordinate | None = None
    language: str | None = None
    filename: str = None
    page_dimensions: list[PageDimensions] = None

    def __init__(self, document: fitz.Document, ground_truth: GroundTruth):
        """Initializes the StratigraphyMetadata object.

        Args:
            document (fitz.Document): A PDF document.
        """
        matching_params = read_params("matching_params.yml")
        ground_truth_metadata = ground_truth.for_file(document.name.split("/")[-1]).get("metadata", {})

        # Detect the language of the document
        self.language = detect_language_of_document(
            document, matching_params["default_language"], matching_params["material_description"].keys()
        )

        # Extract the coordinates of the borehole
        coordinate_extractor = CoordinateExtractor(document=document)
        self.coordinates = coordinate_extractor.extract_coordinates(
            ground_truth_coordinates=ground_truth_metadata["coordinates"]
        )

        # Extract the elevation information
        elevation_extractor = ElevationExtractor(document=document)
        self.elevation = elevation_extractor.extract_elevation(ground_truth_metadata["reference_elevation"])

        # Get the name of the document
        self.filename = document.name

        # Get the dimensions of the document's pages
        self.page_dimensions = []
        for page in document:
            self.page_dimensions.append(PageDimensions(width=page.rect.width, height=page.rect.height))

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "elevation": self.elevation.to_json() if self.elevation else None,
            "coordinates": self.coordinates.to_json() if self.coordinates else None,
            "language": self.language,
            "page_dimensions": [page_dimensions.to_json() for page_dimensions in self.page_dimensions],
        }

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return (
            f"StratigraphyMetadata("
            f"elevation={self.elevation}, "
            f"coordinates={self.coordinates} "
            f"language={self.language}, "
            f"page_dimensions={self.page_dimensions})"
        )

    def evaluate(self, ground_truth_path: str) -> FileBoreholeMetadataMetrics:
        """Evaluate the metadata of the file against the ground truth.

        Args:
            ground_truth_path (str): The path to the ground truth file.
        """
        metadata_ground_truth = GroundTruth(ground_truth_path).for_file(self.filename).get("metadata", {})

        # Initialize the metadata correctness metrics
        coordinate_metrics = None
        elevation_metrics = None

        ############################################################################################################
        ### Compute the metadata correctness for the coordinates.
        ############################################################################################################
        extracted_coordinates = self.coordinates
        ground_truth_coordinates = metadata_ground_truth.get("coordinates")

        if extracted_coordinates and ground_truth_coordinates:
            if extracted_coordinates.east.coordinate_value > 2e6 and ground_truth_coordinates["E"] < 2e6:
                ground_truth_east = int(ground_truth_coordinates["E"]) + 2e6
                ground_truth_north = int(ground_truth_coordinates["N"]) + 1e6
            elif extracted_coordinates.east.coordinate_value < 2e6 and ground_truth_coordinates["E"] > 2e6:
                ground_truth_east = int(ground_truth_coordinates["E"]) - 2e6
                ground_truth_north = int(ground_truth_coordinates["N"]) - 1e6
            else:
                ground_truth_east = int(ground_truth_coordinates["E"])
                ground_truth_north = int(ground_truth_coordinates["N"])

            if (math.isclose(int(extracted_coordinates.east.coordinate_value), ground_truth_east, abs_tol=2)) and (
                math.isclose(int(extracted_coordinates.north.coordinate_value), ground_truth_north, abs_tol=2)
            ):
                coordinate_metrics = Metrics(
                    tp=1,
                    fp=0,
                    fn=0,
                    feature_name="coordinates",
                )
            else:
                coordinate_metrics = Metrics(
                    tp=0,
                    fp=1,
                    fn=1,
                    feature_name="coordinates",
                )
        else:
            coordinate_metrics = Metrics(
                tp=0,
                fp=1 if extracted_coordinates is not None else 0,
                fn=1 if ground_truth_coordinates is not None else 0,
                feature_name="coordinates",
            )

        ############################################################################################################
        ### Compute the metadata correctness for the elevation.
        ############################################################################################################
        extracted_elevation = None if self.elevation is None else self.elevation.elevation
        ground_truth_elevation = metadata_ground_truth.get("reference_elevation")

        if extracted_elevation is not None and ground_truth_elevation is not None:
            if math.isclose(extracted_elevation, ground_truth_elevation, abs_tol=0.1):
                elevation_metrics = Metrics(
                    tp=1,
                    fp=0,
                    fn=0,
                    feature_name="elevation",
                )
            else:
                elevation_metrics = Metrics(
                    tp=0,
                    fp=1,
                    fn=1,
                    feature_name="elevation",
                )
        else:
            elevation_metrics = Metrics(
                tp=0,
                fp=1 if extracted_elevation is not None else 0,
                fn=1 if ground_truth_elevation is not None else 0,
                feature_name="elevation",
            )

        return FileBoreholeMetadataMetrics(
            elevation_metrics=elevation_metrics, coordinates_metrics=coordinate_metrics, filename=self.filename
        )
