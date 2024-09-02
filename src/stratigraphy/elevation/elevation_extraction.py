"""Module for extracting elevation data from text.

This module provides classes and functions to extract elevation data from text. The main class is the
ElevationExtractor class which is used to extract elevation data from text. The class is designed to be
used in combination with the StratigraphyExtractor class to extract elevation data from stratigraphic
descriptions.
"""

import abc
import logging
from dataclasses import dataclass

import fitz
import numpy as np
from stratigraphy.data_extractor.data_extractor import DataExtractor
from stratigraphy.groundwater.utility import extract_elevation
from stratigraphy.util.line import TextLine

logger = logging.getLogger(__name__)


@dataclass
class ElevationInformation(metaclass=abc.ABCMeta):
    """Abstract class for Groundwater Information."""

    elevation: float | None = None  # Elevation of the groundwater relative to the mean sea level
    rect: fitz.Rect | None = None  # The rectangle that contains the extracted information
    page: int | None = None  # The page number of the PDF document

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

    def to_dict(self) -> dict:
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
    the location of a coordinate key in a string of text.
    """

    def get_feature_near_key(self, lines: list[TextLine], page: int, page_width: float) -> list[float]:
        """Find elevation from text lines that are close to an explicit "elevation" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document
            page_width (float): the width of the current page (in points / PyMuPDF coordinates)

        Returns:
            list[float]: all found elevations
        """
        # find the key that indicates the coordinate information
        elevation_key_lines = self.find_feature_key(lines)
        if elevation_key_lines is None:
            return []

        extracted_elevation_informations = []

        for elevation_key_line in elevation_key_lines:
            # find the lines of the text that are close to an identified coordinate key.
            key_rect = elevation_key_line.rect
            # look for coordinate values to the right and/or immediately below the key
            elevation_search_rect = fitz.Rect(
                key_rect.x0, key_rect.y0, key_rect.x1 + 5 * key_rect.width, key_rect.y1 + 1 * key_rect.width
            )
            elevation_lines = [line for line in lines if line.rect.intersects(elevation_search_rect)]

            def preprocess(value: str) -> str:
                value = value.replace(",", ".")
                value = value.replace("'", ".")
                value = value.replace("o", "0")  # frequent ocr error
                value = value.replace("\n", " ")
                value = value.replace("ate", "ote")  # frequent ocr error
                return value

            # makes sure the line with the key is included first in the extracted information and the duplicate removed
            elevation_lines.insert(0, elevation_key_line)
            elevation_lines = list(dict.fromkeys(elevation_lines))

            try:
                extracted_elevation_information = self.get_elevation_from_lines(elevation_lines, page, preprocess)
                if extracted_elevation_information.elevation:
                    extracted_elevation_informations.append(extracted_elevation_information)
            except ValueError as error:
                logger.warning(f"ValueError: {error}")
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
        if extracted_elevation_informations:
            return extracted_elevation_informations[0]
        else:
            return None

    @staticmethod
    def get_elevation_from_lines(lines: list[TextLine], page: int, preprocess=lambda x: x) -> list[float]:
        r"""Matches the coordinates in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary string of text.
            page (int): the page number (1-based) of the PDF document
            preprocess: function that takes a string and returns a preprocessed string  # TODO add type

        Returns:
            list[float]: A list of potential elevation
        """
        matched_lines_rect = []

        for line in lines:
            text = preprocess(line.text)

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
