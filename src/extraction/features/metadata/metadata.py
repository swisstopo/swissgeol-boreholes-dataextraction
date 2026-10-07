"""Metadata for stratigraphy data."""

from dataclasses import dataclass
from typing import NamedTuple

import pymupdf
from pydantic import BaseModel

from extraction.features.metadata.borehole_name_extraction import BoreholeName
from extraction.features.metadata.coordinate_extraction import Coordinate, CoordinateExtractor
from extraction.features.metadata.elevation_extraction import Elevation, ElevationExtractor
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage
from swissgeol_doc_processing.utils.language_detection import detect_language_of_document


class PageDimensions(NamedTuple):
    """Class for page dimensions."""

    width: float
    height: float


@dataclass
class MetadataInDocument:
    """Container for all stratigraphy metadata found in the document."""

    elevations: list[FeatureOnPage[Elevation]]
    coordinates: list[FeatureOnPage[Coordinate]]

    @classmethod
    def from_document(cls, document: pymupdf.Document, language: str, matching_params: dict) -> "MetadataInDocument":
        """Create a MetadataInDocument object from a document.

        Args:
            document (pymupdf.Document): The document.
            language (str): The language of the document.
            matching_params (dict): The matching parameters.

        Returns:
            MetadataInDocument: The metadata object.
        """
        # Extract the coordinates of the borehole
        coordinate_extractor = CoordinateExtractor(language, matching_params)
        coordinates = coordinate_extractor.extract_coordinates(document=document)

        # Extract the elevation information
        elevation_extractor = ElevationExtractor(language, matching_params)
        elevations = elevation_extractor.extract_elevation(document=document)

        return cls(elevations=elevations, coordinates=coordinates)


class BoreholeMetadata(BaseModel):
    """Metadata for stratigraphy data for a single borehole."""

    elevation: FeatureOnPage[Elevation] | None = None
    coordinates: FeatureOnPage[Coordinate] | None = None
    name: FeatureOnPage[BoreholeName] | None = None


class FileMetadata(BaseModel):
    """Class to store and extract metadata at the file level (common to all boreholes in the file)."""

    language: str | None  # TODO: Change to Enum for the supported languages
    page_dimensions: list[PageDimensions]

    @classmethod
    def from_document(cls, document: pymupdf.Document, matching_params: dict) -> "FileMetadata":
        """Create a FileMetadata object from a document.

        Args:
            document (pymupdf.Document): The document.
            matching_params (dict): The matching parameters.

        Returns:
            FileMetadata: The file metadata object.
        """
        # Detect the language of the document
        language = detect_language_of_document(
            document, matching_params["default_language"], matching_params["material_description"].keys()
        )

        # Get the dimensions of the document's pages
        page_dimensions = []
        for page in document:
            page_dimensions.append(PageDimensions(width=page.rect.width, height=page.rect.height))

        # Sanity check
        assert len(page_dimensions) == document.page_count, "Page count mismatch."

        return cls(
            language=language,
            page_dimensions=page_dimensions,
        )
