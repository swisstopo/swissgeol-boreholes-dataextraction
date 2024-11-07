"""This module contains the GroundwaterLevelExtractor class."""

import logging
import os
from dataclasses import dataclass
from datetime import date as dt
from datetime import datetime
from pathlib import Path

import fitz
import numpy as np
from stratigraphy.data_extractor.data_extractor import DataExtractor, ExtractedFeature, FeatureOnPage
from stratigraphy.groundwater.utility import extract_date, extract_depth, extract_elevation
from stratigraphy.lines.line import TextLine
from stratigraphy.metadata.elevation_extraction import Elevation
from stratigraphy.text.extract_text import extract_text_lines

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d"
MAX_DEPTH = 200  # Maximum depth of the groundwater in meters - Otherwise, depth might be confused with
# elevation from the extraction algorithm.
# TODO: One could use the depth column to find the maximal depth of the borehole and use this as a threshold.


@dataclass
class Groundwater(ExtractedFeature):
    """Abstract class for Groundwater Information."""

    depth: float  # Depth of the groundwater relative to the surface
    date: dt | None = (
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
        return f"Groundwater(" f"date={self.format_date()}, " f"depth={self.depth}, " f"elevation={self.elevation})"

    @staticmethod
    def from_json_values(depth: float | None, date: str | None, elevation: float | None) -> "Groundwater":
        """Converts the object from a dictionary.

        Args:
            depth (float | None): The depth of the groundwater.
            date (str | None): The measurement date of the groundwater.
            elevation (float | None): The elevation of the groundwater.

        Returns:
            Groundwater: The object created from the dictionary.
        """
        date = datetime.strptime(date, DATE_FORMAT).date() if date is not None and date != "" else None
        return Groundwater(depth=depth, date=date, elevation=elevation)

    @classmethod
    def from_json(cls, json: dict) -> "Groundwater":
        """Converts a dictionary to an object.

        Args:
            json (dict): A dictionary representing the groundwater information.

        Returns:
            Groundwater: The groundwater information object.
        """
        return cls.from_json_values(
            depth=json["depth"],
            date=json["date"],
            elevation=json["elevation"],
        )

    def format_date(self) -> str | None:
        """Formats the date of the groundwater measurement.

        Returns:
            str | None: The formatted date of the groundwater measurement.
        """
        if self.date is not None:
            return self.date.strftime(DATE_FORMAT)
        else:
            return None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "date": self.format_date(),
            "depth": self.depth,
            "elevation": self.elevation,
        }


@dataclass
class GroundwaterInDocument:
    """Class for extracted groundwater information from a document."""

    groundwater: list[FeatureOnPage[Groundwater]]
    filename: str

    @classmethod
    def from_document(cls, doc: fitz.Document, terrain_elevation: Elevation | None = None) -> "GroundwaterInDocument":
        """Initializes the GroundwaterInDocument object and extracts the groundwater from the document.

        Args:
            doc (fitz.Document): The PDF document.
            terrain_elevation (Elevation | None): The elevation of the terrain.

        Returns:
            GroundwaterInDocument: The extracted groundwater information from the document.
        """
        filename = Path(doc.name).name

        groundwater_extractor = GroundwaterLevelExtractor(document=doc)
        groundwater: list[FeatureOnPage[Groundwater]] = groundwater_extractor.extract_groundwater(terrain_elevation)

        return GroundwaterInDocument(groundwater=groundwater, filename=filename)

    def get_groundwater_per_page(self) -> list[FeatureOnPage[Groundwater]]:
        """Returns the groundwater information in the document.

        Returns:
            list[FeatureOnPage[Groundwater]]: The groundwater information in the document.
        """
        return self.groundwater

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.groundwater]


class GroundwaterLevelExtractor(DataExtractor):
    """Extracts coordinates from a PDF document."""

    feature_name = "groundwater"

    is_searching_groundwater_illustration: bool = False

    # look for elevation values to the left, right and/or immediately below the key
    search_left_factor: float = 2
    search_right_factor: float = 10
    search_below_factor: float = 4
    search_above_factor: float = 0

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ü": "u"}

    def __init__(self, document):
        super().__init__(document)

        self.is_searching_groundwater_illustration = os.getenv("IS_SEARCHING_GROUNDWATER_ILLUSTRATION") == "True"
        if self.is_searching_groundwater_illustration:
            logger.info("Searching for groundwater information in illustrations.")

    def get_groundwater_near_key(self, lines: list[TextLine], page: int) -> list[FeatureOnPage[Groundwater]]:
        """Find groundwater information from text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[FeatureOnPage[Groundwater]]: all found groundwater information
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
                if extracted_gw.feature.depth or extracted_gw.feature.elevation:
                    # if the depth or elevation is extracted, add the extracted groundwater information to the list
                    extracted_groundwater_list.append(extracted_gw)
            except ValueError as error:
                logger.warning("ValueError: %s", error)
                logger.warning("Could not extract groundwater information from the lines near the key.")

        return extracted_groundwater_list

    def get_groundwater_info_from_lines(self, lines: list[TextLine], page: int) -> FeatureOnPage[Groundwater]:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page (int): the page number (1-based) of the PDF document
        Returns:
            FeatureOnPage[Groundwater]: the extracted groundwater information
        """
        date: dt | None = None
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
                    date = extracted_date

                depth = extract_depth(text, MAX_DEPTH)
                if depth:
                    text = text.replace(str(depth), "").strip()

                elevation = extract_elevation(text)

                matched_lines_rect.append(line.rect)
            else:
                # Pattern for matching date
                if not date:
                    extracted_date, extracted_date_str = extract_date(text)
                    if extracted_date_str:
                        text = text.replace(extracted_date_str, "").strip()
                        date = extracted_date
                        matched_lines_rect.append(
                            line.rect
                        )  # Add the rectangle of the line to the matched lines list to make sure it is drawn
                        # in the output image.
                else:
                    # If a second date is present in the lines around the groundwater key, then we skip this line,
                    # instead of potentially falsely extracting a depth value from the date.
                    extracted_date, extracted_date_str = extract_date(text)
                    if extracted_date_str:
                        continue

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
            if date and depth and elevation:
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
        if depth or elevation:
            return FeatureOnPage(
                feature=Groundwater(depth=depth, date=date, elevation=elevation),
                rect=rect_union,
                page=page,
            )
        else:
            raise ValueError("Could not extract all required information from the lines provided.")

    def extract_groundwater(self, terrain_elevation: Elevation | None) -> list[FeatureOnPage[Groundwater]]:
        """Extracts the groundwater information from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.
        Algorithm description:
            1. if that gives no results, search for coordinates close to an explicit "groundwater" label (e.g. "Gswp")

        Args:
            terrain_elevation (ElevationInformation | None): The elevation of the borehole.

        Returns:
            list[FeatureOnPage[Groundwater]]: the extracted coordinates (if any)
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # NOTE: page.number is 0-based

            found_groundwater = self.get_groundwater_near_key(lines, page_number)
            if not found_groundwater and self.is_searching_groundwater_illustration:
                from stratigraphy.groundwater.gw_illustration_template_matching import (
                    get_groundwater_from_illustration,
                )

                # Extract groundwater from illustration
                found_groundwater, confidence_list = get_groundwater_from_illustration(
                    self, lines, page_number, terrain_elevation
                )
                if found_groundwater:
                    logger.info("Confidence list: %s", confidence_list)
                    logger.info("Found groundwater from illustration on page %s: %s", page_number, found_groundwater)

            if terrain_elevation:
                # If the elevation is provided, calculate the depth of the groundwater
                for entry in found_groundwater:
                    if not entry.feature.depth and entry.feature.elevation:
                        entry.feature.depth = round(terrain_elevation.elevation - entry.feature.elevation, 2)
                    if not entry.feature.elevation and entry.feature.depth:
                        entry.feature.elevation = round(terrain_elevation.elevation - entry.feature.depth, 2)

            if found_groundwater:
                groundwater_output = ", ".join([str(entry.feature) for entry in found_groundwater])
                logger.info("Found groundwater information on page %s: %s", page_number, groundwater_output)
                return found_groundwater

        logger.info("No groundwater found in this borehole profile.")
        return []
