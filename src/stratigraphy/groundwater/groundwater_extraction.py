"""This module contains the GroundwaterLevelExtractor class."""

import abc
import logging
from dataclasses import dataclass
from datetime import date, datetime

import fitz
import numpy as np
from stratigraphy.data_extractor.data_extractor import DataExtractor, ExtractedFeature
from stratigraphy.groundwater.utility import extract_date, extract_depth, extract_elevation
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d"
MAX_DEPTH = 200  # Maximum depth of the groundwater in meters - Otherwise, depth might be confused with
# elevation from the extraction algorithm.
# TODO: One could use the depth column to find the maximal depth of the borehole and use this as a threshold.


@dataclass
class GroundwaterInformation(metaclass=abc.ABCMeta):
    """Abstract class for Groundwater Information."""

    depth: float  # Depth of the groundwater relative to the surface
    measurement_date: date | None = (
        None  # Date of the groundwater measurement, if several dates
        # are present, the date of the document the last measurement is taken
    )
    elevation: float | None = None  # Elevation of the groundwater relative to the mean sea level

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
        return (
            f"GroundwaterInformation("
            f"measurement_date={self.format_measurement_date()}, "
            f"depth={self.depth}, "
            f"elevation={self.elevation})"
        )

    @staticmethod
    def from_json_values(depth: float | None, measurement_date: str | None, elevation: float | None):
        if measurement_date is not None and measurement_date != "":
            # convert to datetime object
            measurement_date = datetime.strptime(measurement_date, DATE_FORMAT).date()
        else:
            measurement_date = None

        return GroundwaterInformation(depth=depth, measurement_date=measurement_date, elevation=elevation)

    def format_measurement_date(self) -> str | None:
        if self.measurement_date is not None:
            return self.measurement_date.strftime(DATE_FORMAT)
        else:
            return None


@dataclass(kw_only=True)
class GroundwaterInformationOnPage(ExtractedFeature):
    """Abstract class for Groundwater Information."""

    groundwater: GroundwaterInformation

    def is_valid(self) -> bool:
        """Checks if the information is valid.

        Returns:
            bool: True if the information is valid, otherwise False.
        """
        return self.groundwater > 0

    def to_dict(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "measurement_date": self.groundwater.format_measurement_date(),
            "depth": self.groundwater.depth,
            "elevation": self.groundwater.elevation,
            "page": self.page if self.page else None,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
        }

    @staticmethod
    def from_json_values(
        measurement_date: str | None, depth: float | None, elevation: float | None, page: int, rect: list[float]
    ):
        """Converts the object from a dictionary.

        Args:
            measurement_date (str | None): The measurement date of the groundwater.
            depth (float | None): The depth of the groundwater.
            elevation (float | None): The elevation of the groundwater.
            page (int): The page number of the PDF document.
            rect (list[float]): The rectangle that contains the extracted information.

        Returns:
            GroundwaterInformationOnPage: The object created from the dictionary.
        """
        return GroundwaterInformationOnPage(
            groundwater=GroundwaterInformation.from_json_values(
                depth=depth, measurement_date=measurement_date, elevation=elevation
            ),
            page=page,
            rect=fitz.Rect(rect),
        )


class GroundwaterLevelExtractor(DataExtractor):
    """Extracts coordinates from a PDF document."""

    feature_name = "groundwater"

    # look for elevation values to the left, right and/or immediately below the key
    search_left_factor: float = 2
    search_right_factor: float = 10
    search_below_factor: float = 3

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ü": "u"}

    def get_groundwater_near_key(self, lines: list[TextLine], page: int) -> list[GroundwaterInformationOnPage]:
        """Find groundwater information from text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[GroundwaterInformationOnPage]: all found groundwater information
        """
        # find the key that indicates the groundwater information
        groundwater_key_lines = self.find_feature_key(lines)
        extracted_groundwater_list = []

        for groundwater_key_line in groundwater_key_lines:
            key_rect = groundwater_key_line.rect
            groundwater_info_lines = self.get_lines_near_key(lines, groundwater_key_line)

            # sort the lines by their proximity to the key line center, compute the distance to the key line center
            key_center = (key_rect.x0 + key_rect.x1) / 2
            groundwater_info_lines.sort(key=lambda line: abs((line.rect.x0 + line.rect.x1) / 2 - key_center))

            try:
                extracted_gw = self.get_groundwater_info_from_lines(groundwater_info_lines, page)
                if extracted_gw.groundwater.depth:
                    extracted_groundwater_list.append(extracted_gw)
            except ValueError as error:
                logger.warning("ValueError: %s", error)
                logger.warning("Could not extract groundwater information from the lines near the key.")

        return extracted_groundwater_list

    def get_groundwater_info_from_lines(self, lines: list[TextLine], page: int) -> GroundwaterInformationOnPage:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page (int): the page number (1-based) of the PDF document
        Returns:
            GroundwaterInformationOnPage: the extracted groundwater information
        """
        datetime_date: date | None = None
        depth: float | None = None
        elevation: float | None = None

        matched_lines_rect = []

        for idx, line in enumerate(lines):
            text = self.preprocess(line.text)

            # The first line is the keyword line that contains the groundwater keyword
            if idx == 0:
                # Check if the keyword line contains the date, depth, and elevation, extract them
                extracted_date, extracted_date_str = extract_date(text)
                if extracted_date_str:
                    text = text.replace(extracted_date_str, "").strip()
                    datetime_date = extracted_date

                depth = extract_depth(text, MAX_DEPTH)
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
                    depth = extract_depth(text, MAX_DEPTH)
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
            return GroundwaterInformationOnPage(
                groundwater=GroundwaterInformation(depth=depth, measurement_date=datetime_date, elevation=elevation),
                rect=rect_union,
                page=page,
            )
        else:
            raise ValueError("Could not extract all required information from the lines provided.")

    def extract_groundwater(self) -> list[GroundwaterInformationOnPage]:
        """Extracts the groundwater information from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.
        Algorithm description:
            1. if that gives no results, search for coordinates close to an explicit "groundwater" label (e.g. "Gswp")

        Returns:
            list[GroundwaterInformationOnPage]: the extracted coordinates (if any)
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # page.number is 0-based

            found_groundwater = (
                self.get_groundwater_near_key(lines, page_number)
                # or XXXX # Add other techniques here
            )

            if found_groundwater:
                groundwater_output = ", ".join([str(entry.groundwater) for entry in found_groundwater])
                logger.info("Found groundwater information on page %s: %s", page_number, groundwater_output)
                return found_groundwater

        logger.info("No groundwater found in this borehole profile.")
        return []
