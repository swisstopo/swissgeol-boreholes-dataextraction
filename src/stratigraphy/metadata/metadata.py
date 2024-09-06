"""Metadata for stratigraphy data."""

import abc
from dataclasses import dataclass
from typing import NamedTuple

import fitz
from stratigraphy.metadata.coordinates.coordinate_extraction import Coordinate, CoordinateExtractor
from stratigraphy.metadata.elevation.elevation_extraction import ElevationExtractor, ElevationInformation
from stratigraphy.util.language_detection import detect_language_of_document
from stratigraphy.util.util import read_params


class PageDimensions(NamedTuple):
    """Class for page dimensions."""

    width: float
    height: float

    def to_dict(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {"width": self.width, "height": self.height}


@dataclass
class StratigraphyMetadata(metaclass=abc.ABCMeta):
    """Metadata for stratigraphy data."""

    elevation: ElevationInformation | None = None
    coordinates: Coordinate | None = None
    language: str | None = None
    filename: str = None
    page_dimensions: list[PageDimensions] = None

    def __init__(self, document: fitz.Document):
        """Initializes the StratigraphyMetadata object.

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
        self.filename = document.name

        # Get the dimensions of the document's pages
        self.page_dimensions = []
        for page in document:
            self.page_dimensions.append(PageDimensions(width=page.rect.width, height=page.rect.height))

    def to_dict(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "elevation": self.elevation.to_json() if self.elevation else None,
            "coordinates": self.coordinates.to_json() if self.coordinates else None,
        }

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"StratigraphyMetadata(" f"elevation={self.elevation}, " f"coordinates={self.coordinates})"
