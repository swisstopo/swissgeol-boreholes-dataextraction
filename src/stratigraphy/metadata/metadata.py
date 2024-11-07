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

    def __init__(
        self,
        language: str = None,
        elevation: Elevation = None,
        coordinates: Coordinate = None,
        page_dimensions: list[PageDimensions] = None,
        filename: Path = None,
    ):
        """Initializes the BoreholeMetadata object.

        Args:
            Args:
            language (str | None): The language of the document.
            elevation (Elevation | None): The elevation information.
            coordinates (Coordinate | None): The coordinates of the borehole.
            page_dimensions (list[PageDimensions] | None): The dimensions of the pages in the document.
            filename (Path | None): The name of the file.
        """
        self.language = language
        self.elevation = elevation
        self.coordinates = coordinates
        self.page_dimensions = page_dimensions
        self.filename = filename

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

    @classmethod
    def from_document(cls, document: fitz.Document) -> "BoreholeMetadata":
        """Create a BoreholeMetadata object from a document.

        Args:
            document (fitz.Document): The document.

        Returns:
            BoreholeMetadata: The metadata object.
        """
        matching_params = read_params("matching_params.yml")

        # Detect the language of the document
        language = detect_language_of_document(
            document, matching_params["default_language"], matching_params["material_description"].keys()
        )

        # Extract the coordinates of the borehole
        coordinate_extractor = CoordinateExtractor(document=document)
        coordinates = coordinate_extractor.extract_coordinates()

        # Extract the elevation information
        elevation_extractor = ElevationExtractor(document=document)
        elevation = elevation_extractor.extract_elevation()

        # Get the name of the document
        filename = Path(document.name)

        # Get the dimensions of the document's pages
        page_dimensions = []
        for page in document:
            page_dimensions.append(PageDimensions(width=page.rect.width, height=page.rect.height))

        # Sanity check
        assert len(page_dimensions) == document.page_count, "Page count mismatch."

        return cls(
            language=language,
            elevation=elevation,
            coordinates=coordinates,
            filename=filename,
            page_dimensions=page_dimensions,
        )

    @classmethod
    def from_json(cls, json_metadata: dict, filename: str) -> "BoreholeMetadata":
        """Converts a dictionary to an object.

        Args:
            json_metadata (dict): A dictionary representing the metadata.
            filename (str): The name of the file.

        Returns:
            BoreholeMetadata: The metadata object.
        """
        elevation = Elevation.from_json(json_metadata["elevation"]) if json_metadata["elevation"] is not None else None
        coordinates = (
            Coordinate.from_json(json_metadata["coordinates"]) if json_metadata["coordinates"] is not None else None
        )
        language = json_metadata["language"]
        page_dimensions = [
            PageDimensions(width=page["width"], height=page["height"]) for page in json_metadata["page_dimensions"]
        ]

        return cls(
            elevation=elevation,
            coordinates=coordinates,
            language=language,
            page_dimensions=page_dimensions,
            filename=Path(filename),
        )


@dataclass
class OverallBoreholeMetadata:
    """Metadata for stratigraphy data.

    This class is a list of BoreholeMetadata objects. Each object corresponds to a
    single file.
    """

    metadata_per_file: list[BoreholeMetadata] = None

    def __init__(self):
        """Initializes the BoreholeMetadataList object."""
        self.metadata_per_file = []

    def add_metadata(self, metadata: BoreholeMetadata) -> None:
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
