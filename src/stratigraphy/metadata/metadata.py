"""Metadata for stratigraphy data."""

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import fitz
from stratigraphy.metadata.coordinate_extraction import Coordinate, CoordinateExtractor
from stratigraphy.metadata.elevation_extraction import Elevation, ElevationExtractor
from stratigraphy.metadata.language_detection import detect_language_of_document
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
class BoreholeMetadata(metaclass=abc.ABCMeta):
    """Metadata for stratigraphy data."""

    elevation: Elevation | None = None
    coordinates: Coordinate | None = None
    language: str | None = None  # TODO: Change to Enum for the supported languages
    filename: Path = None
    page_dimensions: list[PageDimensions] = None

    def __init__(self, document: fitz.Document):
        """Initializes the BoreholeMetadata object.

        Args:
            document (fitz.Document): A PDF document.
        """
        matching_params = read_params("matching_params.yml")

        # Detect the language of the document
        self.language = detect_language_of_document(
            document, matching_params["default_language"], matching_params["material_description"].keys()
        )

        # Extract the coordinates of the borehole
        coordinate_extractor = CoordinateExtractor(document=document)
        self.coordinates = coordinate_extractor.extract_coordinates()

        # Extract the elevation information
        elevation_extractor = ElevationExtractor(document=document)
        self.elevation = elevation_extractor.extract_elevation()

        # Get the name of the document
        self.filename = Path(document.name)

        # Get the dimensions of the document's pages
        self.page_dimensions = []
        for page in document:
            self.page_dimensions.append(PageDimensions(width=page.rect.width, height=page.rect.height))

        # Sanity check
        assert len(self.page_dimensions) == document.page_count, "Page count mismatch."

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


@dataclass
class OverallBoreholeMetadata(metaclass=abc.ABCMeta):
    """Metadata for stratigraphy data."""

    metadata_per_file: list[BoreholeMetadata] = None

    def __init__(self):
        """Initializes the StratigraphyMetadata object."""
        self.metadata_per_file = []

    def add_metadata(self, metadata: BoreholeMetadata):
        """Add metadata to the list.

        Args:
            metadata (BoreholeMetadata): The metadata to add.
        """
        self.metadata_per_file.append(metadata)

    def get_metadata(self, filename: str) -> BoreholeMetadata:
        """Get the metadata for a specific file.

        Args:
            filename (str): The name of the file.

        Returns:
            BoreholeMetadata: The metadata for the file.
        """
        for metadata in self.metadata_per_file:
            if metadata.filename.name == filename:
                return metadata
        return None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {metadata.filename.name: metadata.to_json() for metadata in self.metadata_per_file}
