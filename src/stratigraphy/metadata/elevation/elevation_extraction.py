"""Module for extracting elevation data from text.

This module provides classes and functions to extract elevation data from text. The main class is the
ElevationExtractor class which is used to extract elevation data from text. The class is designed to be
used in combination with the StratigraphyExtractor class to extract elevation data from stratigraphic
descriptions.
"""

import logging
from dataclasses import dataclass

import fitz
import numpy as np
from stratigraphy.data_extractor.data_extractor import DataExtractor, ExtractedFeature
from stratigraphy.groundwater.utility import extract_elevation
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine

logger = logging.getLogger(__name__)


@dataclass
class ElevationInformation(ExtractedFeature):
    """Abstract class for Elevation Information."""

    elevation: float | None = None  # Elevation relative to the mean sea level

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
        return f"ElevationInformation(" f"elevation={self.elevation}, " f"page={self.page})"

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "elevation": self.elevation,
            "page": self.page if self.page else None,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
        }


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
    search_below_factor: float = 1

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ate": "ote"}

    def get_elevation_near_key(self, lines: list[TextLine], page: int) -> ElevationInformation | None:
        """Find elevation from text lines that are close to an explicit "elevation" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document
            page_width (float): the width of the current page (in points / PyMuPDF coordinates)

        Returns:
            ElevationInformation | None: the found elevation
        """
        # find the key that indicates the elevation information
        elevation_key_lines = self.find_feature_key(lines)
        extracted_elevation_informations = []

        for elevation_key_line in elevation_key_lines:
            elevation_lines = self.get_lines_near_key(lines, elevation_key_line)

            try:
                extracted_elevation_information = self.get_elevation_from_lines(elevation_lines, page)
                if extracted_elevation_information.elevation:
                    extracted_elevation_informations.append(extracted_elevation_information)
            except ValueError as error:
                logger.warning("ValueError: %s", error)
                logger.warning("Could not extract all required information from the lines provided.")

        return self.select_best_elevation_information(extracted_elevation_informations)

    def select_best_elevation_information(
        self, extracted_elevation_informations: list[ElevationInformation]
    ) -> ElevationInformation | None:
        """Select the best elevation information from a list of extracted elevation information.

        Args:
            extracted_elevation_informations (list[ElevationInformation]): A list of extracted elevation information.

        Returns:
            ElevationInformation | None: The best extracted elevation information.
        """
        # Sort the extracted elevation information by elevation with the highest elevation first
        extracted_elevation_informations.sort(key=lambda x: x.elevation, reverse=True)

        # Return the first element of the sorted list
        return extracted_elevation_informations[0] if extracted_elevation_informations else None

    def get_elevation_from_lines(self, lines: list[TextLine], page: int) -> ElevationInformation:
        r"""Matches the elevation in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary string of text.
            page (int): the page number (1-based) of the PDF document

        Returns:
            ElevationInformation: A list of potential elevation
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
            rect_union = fitz.Rect(x0, y0, x1, y1)
        else:
            rect_union = None

        if elevation:
            return ElevationInformation(elevation=elevation, rect=rect_union, page=page)
        else:
            raise ValueError("Could not extract all required information from the lines provided.")

    def extract_elevation(self) -> ElevationInformation | None:
        """Extracts the elevation information from a borehole profile.

        Processes the borehole profile page by page and tries to find the feature key in the respective text of the
        page.
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # page.number is 0-based

            found_elevation_value = (
                self.get_elevation_near_key(lines, page_number)
                # or XXXX # Add other techniques here
            )

            if found_elevation_value:
                logger.info("Found elevation on page %s: %s", page_number, found_elevation_value.elevation)
                return found_elevation_value

        logger.info("No elevation found in this borehole profile.")
        return None
