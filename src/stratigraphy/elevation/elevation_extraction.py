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
import regex
from stratigraphy.groundwater.utility import extract_elevation
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

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


class ElevationExtractor:
    """Class for extracting elevation data from text.

    The ElevationExtractor class is used to extract elevation data from text. The class is designed
    to be used in combination with the StratigraphyExtractor class to extract elevation data from
    stratigraphic descriptions.

    The class provides methods to extract elevation data from text. The main method is the extract_elevation
    method which extracts elevation data from a list of text lines. The class also provides methods to find
    the location of a coordinate key in a string of text.
    """

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.elevation_keys = read_params("matching_params.yml")["elevation_keys"]

    def find_elevation_key(self, lines: list[TextLine], allowed_errors: int = 3) -> TextLine | None:  # noqa: E501
        """Finds the location of a coordinate key in a string of text.

        This is useful to reduce the text within which the coordinates are searched. If the text is too large
        false positive (found coordinates that are no coordinates) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            TextLine | None: The line of the coordinate key found in the text.
        """
        matches = []
        for key in self.elevation_keys:
            if len(key) < 5:
                # if the key is very short, do an exact match
                pattern = regex.compile(r"\b" + key + r"\b", flags=regex.IGNORECASE)
            else:
                pattern = regex.compile(r"\b" + key + "{e<" + str(allowed_errors) + r"}\b", flags=regex.IGNORECASE)
            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.append((line, sum(match.fuzzy_counts)))

        # if no match was found, return None
        if len(matches) == 0:
            return None

        best_match = min(matches, key=lambda x: x[1])
        return best_match[0]

    def get_elevation_near_key(self, lines: list[TextLine], page: int) -> list[float]:
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
        elevation_key_line = self.find_elevation_key(lines)
        if elevation_key_line is None:
            return []

        # find the lines of the text that are close to an identified coordinate key.
        key_rect = elevation_key_line.rect
        # look for coordinate values to the right and/or immediately below the key
        elevation_search_rect = fitz.Rect(key_rect.x0, key_rect.y0, key_rect.x1 + 5 * key_rect.width, key_rect.y1)
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
            return self.get_elevation_from_lines(elevation_lines, page, preprocess)
        except ValueError:
            logger.warning("Could not extract all required information from the lines provided.")
            return []

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

    def extract_elevation_information(self) -> ElevationInformation | None:
        """Extracts the groundwater information from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.
        Algorithm description:
            1. if that gives no results, search for coordinates close to an explicit "groundwater" label (e.g. "Gswp")

        Returns:
            GroundwaterInformation | None: the extracted coordinates (if any)
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # page.number is 0-based

            found_groundwater_information = (
                self.get_elevation_near_key(lines, page_number)
                # or XXXX # Add other techniques here
            )

            if found_groundwater_information:
                logger.info(f"Found elevation information on page {page_number}: {found_groundwater_information}")
                return found_groundwater_information

        logger.info("No elevation information found in this borehole profile.")
        return None
