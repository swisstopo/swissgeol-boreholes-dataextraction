"""Module for extracting elevation data from text.

This module provides classes and functions to extract elevation data from text. The main class is the
ElevationExtractor class which is used to extract elevation data from text. The class is designed to be
used in combination with the StratigraphyExtractor class to extract elevation data from stratigraphic
descriptions.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pymupdf
from extraction.features.groundwater.utility import extract_elevation
from extraction.features.utils.data_extractor import (
    DataExtractor,
    ExtractedFeature,
    FeatureOnPage,
)
from extraction.features.utils.text.extract_text import extract_text_lines_from_bbox
from extraction.features.utils.text.textline import TextLine

logger = logging.getLogger(__name__)


@dataclass
class Elevation(ExtractedFeature):
    """Abstract class for Elevation Information."""

    elevation: float  # Elevation relative to the mean sea level

    def __post_init__(self):
        """Checks if the information is valid."""
        if not isinstance(self.elevation, float):
            raise ValueError("Elevation must be a float")

    def is_valid(self) -> bool:
        """Checks if the information is valid.

        Returns:
            bool: True if the information is valid, otherwise False.
        """
        return self.elevation > 0

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Elevation(elevation={self.elevation})"

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {"elevation": self.elevation}

    @classmethod
    def from_json(cls, data: dict) -> "Elevation":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the elevation information.

        Returns:
            Elevation: The elevation information object.
        """
        return cls(elevation=data["elevation"])


class ElevationExtractor(DataExtractor):
    """Class for extracting elevation data from text.

    The ElevationExtractor class is used to extract elevation data from text. The class is designed
    to be used in combination with the StratigraphyExtractor class to extract elevation data from
    stratigraphic descriptions.

    The class provides methods to extract elevation data from text. The main method is the extract_elevation
    method which extracts elevation data from a list of text lines. The class also provides methods to find
    the location of an elevation key in a string of text.
    """

    feature_name = "elevation"

    # look for elevation values to the right and/or immediately below the key
    search_right_factor: float = 5
    search_below_factor: float = 4

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ate": "ote"}

    def get_elevation_near_key(self, lines: list[TextLine], page: int) -> list[FeatureOnPage[Elevation]]:
        """Find elevation from text lines that are close to an explicit "elevation" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[FeatureOnPage[Elevation]]: the found elevation
        """
        # find the key that indicates the elevation information
        elevation_key_lines = self.find_feature_key(lines)
        extracted_elevation_list = []

        for elevation_key_line in elevation_key_lines:
            elevation_lines = self.get_lines_near_key(lines, elevation_key_line)  # Check the sorting of the lines

            try:
                extracted_elevation = self.get_elevation_from_lines(elevation_lines, page)
                if extracted_elevation.feature.elevation and extracted_elevation not in extracted_elevation_list:
                    extracted_elevation_list.append(extracted_elevation)
            except ValueError as error:
                logger.warning("ValueError: %s", error)
                logger.warning("Could not extract all required information from the lines provided.")

        return self.select_best_elevation_information(extracted_elevation_list)

    def select_best_elevation_information(
        self, extracted_elevation_list: list[FeatureOnPage[Elevation]]
    ) -> list[FeatureOnPage[Elevation]]:
        """Select the best elevations information from a list of extracted elevation information.

        Currently returns all of the elevations found, sorted with the highest first.

        Args:
            extracted_elevation_list (list[FeatureOnPage[Elevation]]): A list of extracted elevation information.

        Returns:
            list[FeatureOnPage[Elevation]]: The best extracted elevation information.
        """
        # Sort the extracted elevation information by elevation with the highest elevation first
        extracted_elevation_list.sort(key=lambda x: x.feature.elevation, reverse=True)

        # Return all elements of the sorted list
        return extracted_elevation_list

    def get_elevation_from_lines(self, lines: list[TextLine], page: int) -> FeatureOnPage[Elevation]:
        r"""Matches the elevation in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary string of text.
            page (int): the page number (1-based) of the PDF document

        Returns:
            Elevation: A list of potential elevation
        """
        matched_lines_rect = []

        for line in lines:
            text = self.preprocess(line.text)

            # Check if the keyword line contains the elevation, extract it
            elevation = extract_elevation(text)
            if elevation:
                # Pattern for matching depth (e.g., "1,48 m u.T.")
                matched_lines_rect.append(line.rect)
                break

        # Get the union of all matched lines' rectangles
        if len(matched_lines_rect) > 0:
            # make sure the rectangles are unique - As some lines can contain both date and depth
            unique_matched_lines_rect = []
            for rect in matched_lines_rect:
                if rect not in unique_matched_lines_rect:
                    unique_matched_lines_rect.append(rect)

            rect_array = np.array(unique_matched_lines_rect)
            x0 = rect_array[:, 0].min()
            y0 = rect_array[:, 1].min()
            x1 = rect_array[:, 2].max()
            y1 = rect_array[:, 3].max()
            rect_union = pymupdf.Rect(x0, y0, x1, y1)
        else:
            rect_union = None

        if elevation:
            return FeatureOnPage(feature=Elevation(elevation=elevation), rect=rect_union, page=page)
        else:
            raise ValueError("Could not extract all required information from the lines provided.")

    def extract_elevation_from_bbox(
        self, pdf_page: pymupdf.Page, page_number: int, bbox: pymupdf.Rect | None = None
    ) -> list[FeatureOnPage[Elevation]]:
        """Extract the elevation information from a bounding box.

        Args:
            pdf_page (pymupdf.Page): The PDF page.
            bbox (pymupdf.Rect | None): The bounding box.
            page_number (int): The page number.

        Returns:
            list[FeatureOnPage[Elevation]]: The extracted elevation information.
        """
        lines = extract_text_lines_from_bbox(pdf_page, bbox)

        found_elevation_values = self.get_elevation_near_key(lines, page_number)

        if found_elevation_values:
            logger.info(
                "Found elevations in the bounding box: %s",
                [value.feature.elevation for value in found_elevation_values],
            )
        else:
            logger.info("No elevation found in the bounding box.")

        return found_elevation_values

    def extract_elevation(self, document: pymupdf.Document) -> list[FeatureOnPage[Elevation]]:
        """Extracts the elevation information from a borehole profile.

        Processes the borehole profile page by page and tries to find the feature key in the respective text of the
        page.

        Args:
            document (pymupdf.Document): document from which elevation is extracted page by page

        Returns:
            list[FeatureOnPage[Elevation]]: The extracted elevation information.
        """
        elevations: list[Elevation] = []
        for page in document:
            page_number = page.number + 1  # page.number is 0-based
            elevations.extend(self.extract_elevation_from_bbox(page, page_number))

        return elevations
