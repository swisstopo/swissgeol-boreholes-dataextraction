"""Metadata for stratigraphy data."""

from dataclasses import dataclass
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
class BoreholeMetadata:
    """Metadata for stratigraphy data."""

    elevation: FeatureOnPage[Elevation] | None = None
    coordinates: FeatureOnPage[Coordinate] | None = None
    language: str | None = None  # TODO: Change to Enum for the supported languages
    filename: Path = None
    page_dimensions: list[PageDimensions] = None

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
        coordinate_extractor = CoordinateExtractor()
        coordinates = coordinate_extractor.extract_coordinates(document=document)

        # Extract the elevation information
        elevation_extractor = ElevationExtractor()
        elevation = elevation_extractor.extract_elevation(document=document)

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
class OverallFileMetadata:
    """Metadata for stratigraphy data.

    This class is a list of BoreholeMetadata objects. Each object corresponds to a
    single file.
    """

    metadata_per_file: list[list[BoreholeMetadata]] = None

    def __init__(self, metadata: list[list[BoreholeMetadata]] = None):
        """Initializes the BoreholeMetadataList object."""
        if metadata is None:
            self.metadata_per_file = []
        else:
            self.metadata_per_file = metadata

    def add_metadata(self, borehole_metadata_list: list[BoreholeMetadata]) -> None:
        """Add metadata to the list.

        Args:
            borehole_metadata_list (list[BoreholeMetadata]): The list of metadata of the boreholes in a file to add.
        """
        self.metadata_per_file.append(borehole_metadata_list)

    def get_metadata(self, filename: str) -> list[BoreholeMetadata]:
        """Get the metadatas for a specific file.

        Args:
            filename (str): The name of the file.

        Returns:
            list[BoreholeMetadata]: The metadatas for all the borehole in the file.
        """
        for metadata in self.metadata_per_file:
            # all BoreholeMetadata objects in the list have the same filename
            if metadata[0].filename.name == filename:
                return metadata
        return None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            metadata_list[0].filename.name: [metadata.to_json() for metadata in metadata_list]
            for metadata_list in self.metadata_per_file
        }
