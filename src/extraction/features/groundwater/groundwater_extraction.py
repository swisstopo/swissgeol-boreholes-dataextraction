"""This module contains the GroundwaterLevelExtractor class."""

import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pymupdf
from scipy.stats import pearsonr

from extraction.features.groundwater.groundwater_symbol_detection import get_rects_near_symbol
from extraction.features.groundwater.utility import extract_date, extract_depth, extract_elevation
from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer
from extraction.features.utils.data_extractor import (
    DataExtractor,
    ExtractedFeature,
    FeatureOnPage,
)
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.textline import TextLine
from extraction.features.utils.utility import get_lines_near_rect

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d"
MAX_DEPTH = 200  # Maximum depth of the groundwater in meters - Otherwise, depth might be confused with
# elevation from the extraction algorithm.
# TODO: One could use the depth column to find the maximal depth of the borehole and use this as a threshold.


@dataclass
class Groundwater(ExtractedFeature):
    """Abstract class for Groundwater Information."""

    depth: float  # Depth of the groundwater relative to the surface
    date: datetime.date | None = (
        None  # Date of the groundwater measurement, if several dates
        # are present, the date of the document the last measurement is taken
    )
    elevation: float | None = None  # Elevation of the groundwater relative to the mean sea level

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Groundwater(date={self.format_date()}, depth={self.depth}, elevation={self.elevation})"

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
        date = datetime.datetime.strptime(date, DATE_FORMAT) if date is not None and date != "" else None
        if date > datetime.datetime.now():
            date = date.replace(year=date.year - 100)
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

    def infer_infos(self, terrain_elevation: float | None, layers: list[Layer], feature_rect: pymupdf.Rect):
        """Sets the depth or elevation of the groundwater, knowing one and the terrain elevation.

        If both informations are missing, tries to infer them from the given layers and the feature rectangle.

        Args:
            terrain_elevation (float): The elevation of the terrain at the top of the borehole.
            layers (list[Layer]): The list of layers in the borehole.
            feature_rect (pymupdf.Rect): The bounding box of the groundwater feature.
        """
        if self.depth is None:
            if self.elevation is not None and terrain_elevation is not None:
                self.depth = round(terrain_elevation - self.elevation, 2)
            else:
                self.depth = self.infer_depth(layers, feature_rect)

        if self.depth is None:
            return

        if self.elevation is None and terrain_elevation is not None:
            self.elevation = round(terrain_elevation - self.depth, 2)

    def infer_depth(self, layers: list[Layer], feature_rect: pymupdf.Rect) -> float | None:
        """Infers the depth of the groundwater feature based on the given layers and feature rectangle.

        Args:
            layers (list[Layer]): The list of layers in the borehole.
            feature_rect (pymupdf.Rect): The bounding box of the groundwater feature.

        Returns:
            float | None: The inferred depth of the groundwater feature, or None if it could not be determined.
        """
        # Step 1: Prepare depths and y-values
        depths = np.array(
            sorted(
                {
                    (d.value, (d.rect.y0 + d.rect.y1) / 2)
                    for layer in layers
                    if layer.depths is not None
                    for d in (layer.depths.start, layer.depths.end)
                    if d is not None
                },
                key=lambda d: d[0],
            )
        )
        if depths.size == 0:
            return None

        # Step 2: Compute correlation
        corr, p_val = pearsonr(depths[:, 0], depths[:, 1])
        if corr < 0.95 or p_val > 0.01:
            return None

        # Step 3: fit the linear regression and infer the depth
        y_value = feature_rect.y1  # the groundwater limit is usually bellow the date (or elevation) bounding box.
        if y_value < min(depths[:, 1]) or y_value > max(depths[:, 1]):  # out of bounds, not reliable
            return None
        a, b = np.polyfit(depths[:, 0], depths[:, 1], 1)
        depth = round((y_value - b) / a, 2)
        logger.info(f"Infered depth for groundwater: {depth}")
        return depth


@dataclass
class GroundwatersInBorehole:
    """Class for extracted groundwater information from a single borehole."""

    groundwater_feature_list: list[FeatureOnPage[Groundwater]]

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.groundwater_feature_list]

    @classmethod
    def from_json(cls, json_object: list[dict]) -> "GroundwatersInBorehole":
        """Extract a GroundwatersInBorehole object from a json dictionary.

        Args:
            json_object (list[dict]): the json object containing the informations of the borehole

        Returns:
            GroundwatersInBorehole: the GroundwatersInBorehole object
        """
        return cls([FeatureOnPage.from_json(gw_data, Groundwater) for gw_data in json_object])

    def infer_infos(self, terrain_elevation: float | None, layers: list[Layer]):
        """Sets the depth and elevation of all groundwater entries.

        Args:
            terrain_elevation (float): The elevation of the terrain at the top of the borehole.
            layers (list[Layer]): The list of layers in the borehole.
        """
        for entry in self.groundwater_feature_list:
            entry.feature.infer_infos(terrain_elevation, layers, entry.rect)


@dataclass
class GroundwaterInDocument:
    """Class for extracted groundwater information from a document."""

    groundwater_feature_list: list[FeatureOnPage[Groundwater]]
    filename: str

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.groundwater_feature_list]


class GroundwaterLevelExtractor(DataExtractor):
    """Extract groundwater informations from a PDF document."""

    feature_name = "groundwater"

    # look for elevation values to the left, right and/or immediately below the key
    search_left_factor: float = 2
    search_right_factor: float = 8
    search_below_factor: float = 2
    search_above_factor: float = 0

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ü": "u"}

    def get_text_lines_near_key(self, lines: list[TextLine]) -> list[list[TextLine]]:
        """Extracts the text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in

        Returns:
            list[list[TextLine]]: all found lists of textlines that appeared arround a key
        """
        # find the keys that indicates the groundwater information
        groundwater_key_lines = self.find_feature_key(lines)
        extracted_lines_list = []

        for groundwater_key_line in groundwater_key_lines:
            key_rect = groundwater_key_line.rect
            groundwater_info_lines = self.get_lines_near_key(lines, groundwater_key_line)

            # sort the lines by their proximity to the key line center, compute the distance to the key line center
            key_center = (key_rect.x0 + key_rect.x1) / 2
            extracted_lines_list.append(
                sorted(groundwater_info_lines, key=lambda line: abs((line.rect.x0 + line.rect.x1) / 2 - key_center))
            )
        return extracted_lines_list

    def get_text_lines_near_symbol(
        self,
        lines: list[TextLine],
        geometric_lines: list[Line],
        extracted_boreholes: list[ExtractedBorehole],
    ) -> list[list[TextLine]]:
        """Extracts the text lines that are close to a groundwater symbol.

        Args:
            lines (list[TextLine]): The list of text lines to search.
            geometric_lines (list[Line]): The list of geometric lines to consider.
            extracted_boreholes (list[ExtractedBorehole]): The list of extracted boreholes.

        Returns:
            list[list[TextLine]]: The list of text lines near groundwater symbols.
        """
        seen_depths = [lay.depths for bh in extracted_boreholes for lay in bh.predictions if lay.depths]
        seen_depths = [d for depth in seen_depths for d in (depth.start, depth.end) if d]

        rects = get_rects_near_symbol(lines, geometric_lines, seen_depths)

        groundwater_info_lines = [
            get_lines_near_rect(
                self.search_left_factor,
                self.search_right_factor,
                self.search_above_factor,
                self.search_below_factor,
                lines,
                rect,
            )
            for rect in rects
        ]

        return groundwater_info_lines

    def get_groundwater_infos_from_lines(
        self, lines_list: list[list[TextLine]], page_number: int
    ) -> list[FeatureOnPage[Groundwater]]:
        """Extracts the groundwater information from all the lists of text line identified.

        Args:
            lines_list (list[list[TextLine]]): the list of lines of text to extract the groundwater information from.
            page_number (int): the page number (1-based) of the PDF document
        Returns:
            FeatureOnPage[Groundwater]: the extracted groundwater information
        """
        groundwaters = [self.get_groundwater_from_lines(lines, page_number) for lines in lines_list]
        return [gw for gw in groundwaters if gw is not None]

    def get_groundwater_from_lines(self, lines: list[TextLine], page_number: int) -> FeatureOnPage[Groundwater] | None:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page_number (int): the page number (1-based) of the PDF document
        Returns:
            FeatureOnPage[Groundwater]: the extracted groundwater information
        """
        date: datetime.date | None = None
        depth: float | None = None
        elevation: float | None = None

        matched_lines_rect = []
        for line in lines:
            text = self.preprocess(line.text)

            extracted_date, extracted_date_str = extract_date(text)
            if extracted_date_str and not date:
                date = extracted_date
                text = text.replace(extracted_date_str, "").strip()
                matched_lines_rect.append(line.rect)
            elif extracted_date_str and date:
                continue  # skip extra dates

            depth_val = extract_depth(text, MAX_DEPTH)
            if depth_val and not depth:
                depth = depth_val
                text = text.replace(str(depth), "").strip()
                matched_lines_rect.append(line.rect)

            elevation_val = extract_elevation(text)
            if elevation_val and not elevation:
                elevation = elevation_val
                matched_lines_rect.append(line.rect)

            if date and depth and elevation:
                break

        if not matched_lines_rect:
            return None

        rect_union = matched_lines_rect[0]
        for rect in matched_lines_rect[1:]:
            rect_union |= rect

        # return anyway, we can infer informations later
        return FeatureOnPage(
            feature=Groundwater(depth=depth, date=date, elevation=elevation),
            rect=rect_union,
            page=page_number,
        )

    def filter_duplicates(
        self, found_groundwaters: list[FeatureOnPage[Groundwater]]
    ) -> list[FeatureOnPage[Groundwater]]:
        """Filters out duplicate groundwater features from the list.

        Args:
            found_groundwaters (list[FeatureOnPage[Groundwater]]): The list of found groundwater features.

        Returns:
            list[FeatureOnPage[Groundwater]]: The filtered list of unique groundwater features.
        """
        unique_groundwaters = []
        for gw in found_groundwaters:
            keep = True
            to_remove = []
            for other_gw in unique_groundwaters:
                if gw.rect.intersects(other_gw.rect):
                    if gw.rect.get_area() > other_gw.rect.get_area():
                        to_remove.append(other_gw)  # replace smaller with bigger
                    else:
                        keep = False  # skip gw
            for other_gw in to_remove:
                unique_groundwaters.remove(other_gw)
            if keep:
                unique_groundwaters.append(gw)
        return unique_groundwaters

    def extract_groundwater(
        self,
        page_number: int,
        lines: list[TextLine],
        geometric_lines: list[Line],
        extracted_boreholes: list[ExtractedBorehole],
    ) -> list[FeatureOnPage[Groundwater]]:
        """Extracts the groundwater information from a borehole profile.

        Args:
            page_number (int): The page number (1-indexed) of the PDF document.
            lines (list[TextLine]): The lines of text to extract the groundwater information from.
            geometric_lines (list[Line]): The geometric lines on the page.
            extracted_boreholes (list[ExtractedBorehole]): The extracted boreholes from the page.

        Returns:
            list[FeatureOnPage[Groundwater]]: the extracted coordinates (if any)
        """
        grounwater_lines_list = self.get_text_lines_near_key(lines)
        grounwater_lines_list.extend(self.get_text_lines_near_symbol(lines, geometric_lines, extracted_boreholes))

        found_groundwaters = self.get_groundwater_infos_from_lines(grounwater_lines_list, page_number)

        unique_groundwaters = self.filter_duplicates(found_groundwaters)

        if unique_groundwaters:
            groundwater_output = ", ".join([str(entry.feature) for entry in unique_groundwaters])
            logger.info("Found groundwater information on page %s: %s", page_number, groundwater_output)
            return unique_groundwaters

        logger.info("No groundwater found in this borehole profile.")
        return []
