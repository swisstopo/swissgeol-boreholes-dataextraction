"""Metadata for stratigraphy data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import fitz
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
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
class MetadataInDocument:
    """Container for all stratigraphy metadata found in the document."""

    elevations: list[FeatureOnPage[Elevation]]
    coordinates: list[FeatureOnPage[Coordinate]]

    @classmethod
    def from_document(cls, document: fitz.Document) -> "MetadataInDocument":
        """Create a MetadataInDocument object from a document.

        Args:
            document (fitz.Document): The document.

        Returns:
            MetadataInDocument: The metadata object.
        """
        # Extract the coordinates of the borehole
        coordinate_extractor = CoordinateExtractor()
        coordinates = coordinate_extractor.extract_coordinates(document=document)

        # Extract the elevation information
        elevation_extractor = ElevationExtractor()
        elevations = elevation_extractor.extract_elevation(document=document)

        return cls(elevations=elevations, coordinates=coordinates)


@dataclass
class BoreholeMetadata:
    """Metadata for stratigraphy data for a single borehole."""

    elevation: FeatureOnPage[Elevation] | None = None
    coordinates: FeatureOnPage[Coordinate] | None = None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "elevation": self.elevation.to_json() if self.elevation else None,
            "coordinates": self.coordinates.to_json() if self.coordinates else None,
        }

    @classmethod
    def from_json(cls, json_metadata: dict) -> "BoreholeMetadata":
        """Converts a dictionary to an object.

        Args:
            json_metadata (dict): A dictionary representing the metadata.

        Returns:
            BoreholeMetadata: The metadata object.
        """
        return cls(
            FeatureOnPage.from_json(json_metadata["elevation"], Elevation) if json_metadata["elevation"] else None,
            FeatureOnPage.from_json(json_metadata["coordinates"], Coordinate)
            if json_metadata["coordinates"]
            else None,
        )


@dataclass
class FileMetadata:
    """Class to store and extract metadata at the file level (common to all boreholes in the file)."""

    language: str | None = None  # TODO: Change to Enum for the supported languages
    filename: Path = None
    page_dimensions: list[PageDimensions] = None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "language": self.language,
            "page_dimensions": [page_dimensions.to_json() for page_dimensions in self.page_dimensions],
        }

    @classmethod
    def from_document(cls, document: fitz.Document) -> "FileMetadata":
        """Create a FileMetadata object from a document.

        Args:
            document (fitz.Document): The document.

        Returns:
            FileMetadata: The file metadata object.
        """
        matching_params = read_params("matching_params.yml")

        # Detect the language of the document
        language = detect_language_of_document(
            document, matching_params["default_language"], matching_params["material_description"].keys()
        )

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
            filename=filename,
            page_dimensions=page_dimensions,
        )

    @classmethod
    def from_json(cls, json_metadata: dict, filename: str) -> "FileMetadata":
        """Converts a dictionary to an object.

        Args:
            json_metadata (dict): A dictionary representing the metadata.
            filename (str): The name of the file.

        Returns:
            FileMetadata: The metadata object.
        """
        language = json_metadata["language"]
        page_dimensions = [
            PageDimensions(width=page["width"], height=page["height"]) for page in json_metadata["page_dimensions"]
        ]

        return cls(
            language=language,
            page_dimensions=page_dimensions,
            filename=Path(filename),
        )


@dataclass
class OverallFileMetadata:
    """Metadata for stratigraphy data.

    This class is a list of list of BoreholeMetadata objects. Each inner list corresponds to a
    single file.
    """

    filenames: list[str]
    metadata_per_file: list[list[BoreholeMetadata]] = field(default_factory=list)

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            filename: [metadata.to_json() for metadata in metadata_list]
            for filename, metadata_list in zip(self.filenames, self.metadata_per_file, strict=True)
        }
