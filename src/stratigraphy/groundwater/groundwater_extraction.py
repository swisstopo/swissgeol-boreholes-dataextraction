"""This module contains the GroundwaterLevelExtractor class."""

import abc
import logging
from dataclasses import dataclass
from datetime import date

import fitz
import numpy as np
import regex
from stratigraphy.groundwater.utility import extract_date, extract_depth, extract_elevation
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)


@dataclass
class GroundwaterInformation(metaclass=abc.ABCMeta):
    """Abstract class for Groundwater Information."""

    depth: float  # Depth of the groundwater relative to the surface
    measurement_date: date | None = (
        None  # Date of the groundwater measurement, if several dates
        # are present, the date of the document the last measurement is taken
    )
    elevation: float | None = None  # Elevation of the groundwater relative to the mean sea level
    rect: fitz.Rect | None = None  # The rectangle that contains the extracted information
    page: int | None = None  # The page number of the PDF document

    def is_valid(self) -> bool:
        """Checks if the information is valid.

        Returns:
            bool: True if the information is valid, otherwise False.
        """
        return self.depth > 0

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        measurement_date_str = (
            self.measurement_date.strftime("%d.%m.%Y") if self.measurement_date is not None else None
        )
        return (
            f"GroundwaterInformation("
            f"measurement_date={measurement_date_str}, "
            f"depth={self.depth}, "
            f"elevation={self.elevation}, "
            f"page={self.page})"
        )

    def to_dict(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "measurement_date": self.measurement_date.strftime("%d.%m.%Y")
            if self.measurement_date is not None
            else None,
            "depth": self.depth,
            "elevation": self.elevation,
            "page": self.page if self.page else None,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
        }

    def __eq__(self, other: object) -> bool:
        """Checks if two GroundwaterInformation objects are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, otherwise False.
        """
        if not isinstance(other, GroundwaterInformation):
            return NotImplemented

        assert other.is_valid(), "The extracted information is not valid."

        return (
            self.measurement_date == other.measurement_date
            and self.depth == other.depth
            and self.elevation == other.elevation
        )

    def __hash__(self) -> int:
        """Generates a hash for the GroundwaterInformation object.

        Returns:
            int: The hash value of the object.
        """
        return hash((self.measurement_date, self.depth, self.elevation))


class GroundwaterLevelExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.groundwater_keys = read_params("matching_params.yml")["groundwater_keys"]

    def find_groundwater_key(self, lines: list[TextLine], allowed_errors: int = 3) -> list[TextLine] | None:  # noqa: E501
        """Finds the location of a groundwater key in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            TextLine | None: The line of the drilling method key found in the text.
        """
        matches = []
        for key in self.groundwater_keys:
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

        # Remove duplicates
        matches = list(dict.fromkeys(matches))

        # Sort the matches by their error counts (ascending order)
        matches.sort(key=lambda x: x[1])

        # Return the top three matches (lines only)
        return [match[0] for match in matches[:5]]

    def get_groundwater_near_key(self, lines: list[TextLine], page: int, page_width: float) -> GroundwaterInformation:
        """Find coordinates from text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document
            page_width (float): the width of the current page (in points / PyMuPDF coordinates)

        Returns:
            list[Coordinate]: all found groundwater information
        """
        # find the key that indicates the groundwater information
        groundwater_key_lines = self.find_groundwater_key(lines)
        if groundwater_key_lines is None:
            return []

        extracted_groundwater_informations = []

        for groundwater_key_line in groundwater_key_lines:
            # find the lines of the text that are close to an identified groundwater key.
            key_rect = groundwater_key_line.rect
            # look for groundwater related values to the right and/or immediately below the key
            groundwater_info_search_rect = fitz.Rect(
                max(0, key_rect.x0 - 2.0 * key_rect.width), key_rect.y0, page_width, key_rect.y1 + 3 * key_rect.height
            )
            groundwater_info_lines = [line for line in lines if line.rect.intersects(groundwater_info_search_rect)]

            def preprocess(value: str) -> str:
                value = value.replace(",", ".")
                value = value.replace("'", ".")
                value = value.replace("o", "0")
                value = value.replace("\n", " ")
                value = value.replace("ü", "u")
                return value

            # makes sure the line with the key is included first in the extracted information and the duplicate removed
            groundwater_info_lines.insert(0, groundwater_key_line)
            groundwater_info_lines = list(dict.fromkeys(groundwater_info_lines))

            # sort the lines by their proximity to the key line center, compute the distance to the key line center
            key_center = (key_rect.x0 + key_rect.x1) / 2
            groundwater_info_lines.sort(key=lambda line: abs((line.rect.x0 + line.rect.x1) / 2 - key_center))

            try:
                extracted_gw_information = self.get_groundwater_info_from_lines(
                    groundwater_info_lines, page, preprocess
                )
                if extracted_gw_information.depth:
                    extracted_groundwater_informations.append(extracted_gw_information)
            except ValueError as error:
                logger.warning(f"ValueError: {error}")
                logger.warning("Could not extract groundwater information from the lines near the key.")

        return self.select_best_groundwater_information(extracted_groundwater_informations)

    def select_best_groundwater_information(
        self, extracted_groundwater_informations: list[GroundwaterInformation]
    ) -> GroundwaterInformation | None:
        """Selects the best groundwater information from a list of extracted groundwater information.

        Args:
            extracted_groundwater_informations (List[GroundwaterInformation]): The extracted groundwater information.

        Returns:
            GroundwaterInformation | None: The best groundwater information.
        """
        if len(extracted_groundwater_informations) == 0:
            return None

        # If there is only one extracted groundwater information, return it
        if len(extracted_groundwater_informations) == 1:
            return extracted_groundwater_informations[0]

        # if the date is none remove the extracted groundwater information
        extracted_groundwater_informations_with_date = [
            info for info in extracted_groundwater_informations if info.measurement_date is not None
        ]

        # If there are no extracted groundwater information with date, return the first one
        if len(extracted_groundwater_informations_with_date) == 0:
            return extracted_groundwater_informations[0]

        # If there are multiple extracted groundwater information, return the one with the most recent date
        extracted_groundwater_informations_with_date.sort(key=lambda x: x.measurement_date, reverse=True)

        # remove the one with no elevation
        extracted_groundwater_informations_with_date_and_elevation = [
            info for info in extracted_groundwater_informations_with_date if info.elevation is not None
        ]
        if len(extracted_groundwater_informations_with_date_and_elevation) > 0:
            return extracted_groundwater_informations_with_date_and_elevation[0]

        return extracted_groundwater_informations_with_date[0]

    def get_groundwater_info_from_lines(
        self, lines: list[TextLine], page: int, preprocess: lambda x: x
    ) -> GroundwaterInformation:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page (int): the page number (1-based) of the PDF document
            preprocess (callable[[str], str]): a function to preprocess the text of the lines
        Returns:
            GroundwaterInformation: the extracted groundwater information
        """
        datetime_date: date = None
        depth: float = None
        elevation: float = None

        matched_lines_rect = []

        for idx, line in enumerate(lines):
            text = preprocess(line.text)

            # The first line is the keyword line that contains the groundwater keyword
            if idx == 0:
                # Check if the keyword line contains the date, depth, and elevation, extract them
                extracted_date, extracted_date_str = extract_date(text)
                if extracted_date_str:
                    text = text.replace(extracted_date_str, "").strip()
                    datetime_date = extracted_date

                depth = extract_depth(text)
                if depth:
                    text = text.replace(str(depth), "").strip()

                elevation = extract_elevation(text)

                # Pattern for matching depth (e.g., "1,48 m u.T.")
                matched_lines_rect.append(line.rect)
            else:
                # Pattern for matching date
                if not datetime_date:
                    extracted_date, extracted_date_str = extract_date(text)
                    if extracted_date_str:
                        text = text.replace(extracted_date_str, "").strip()
                        datetime_date = extracted_date

                # Pattern for matching depth (e.g., "1,48 m u.T.")
                if not depth:
                    depth = extract_depth(text)
                    if depth:
                        matched_lines_rect.append(line.rect)
                        text = text.replace(str(depth), "").strip()

                # Pattern for matching elevation (e.g., "457,69 m U.M.")
                if not elevation:
                    elevation = extract_elevation(text)
                    if elevation:
                        matched_lines_rect.append(line.rect)

            # If all required data is found, break early
            if datetime_date and depth and elevation:
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

        # Return the populated data class if all values are found
        # if date and depth and elevation:
        #   # TODO: Make sure the elevation is extracted to add it here
        # if date and depth:  # elevation is optional
        #   # TODO: IF the date is not provided for the groundwater (most of the time because there was only one
        # drilling date - chose the date of the document. Date needs to be extracted from the document separately)
        if depth:
            return GroundwaterInformation(
                depth=depth, measurement_date=datetime_date, elevation=elevation, rect=rect_union, page=page
            )
        else:
            raise ValueError("Could not extract all required information from the lines provided.")

    def extract_groundwater_information(self) -> GroundwaterInformation | None:
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
                self.get_groundwater_near_key(lines, page_number, page.rect.width)
                # or XXXX # Add other techniques here
            )

            if found_groundwater_information:
                logger.info(f"Found groundwater information on page {page_number}: {found_groundwater_information}")
                return found_groundwater_information

        logger.info("No groundwater found in this borehole profile.")
        return None
