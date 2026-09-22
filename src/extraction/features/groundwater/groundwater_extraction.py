"""Module for the automatic extraction of groundwater measurements."""

import datetime
import logging

from extraction.features.extracted_borehole import ExtractedBorehole
from extraction.features.groundwater.groundwater import Groundwater
from extraction.features.groundwater.groundwater_color_detection import get_minority_color_lines
from extraction.features.groundwater.groundwater_symbol_detection import (
    get_groundwater_symbol_upper_lines,
    get_text_lines_near_symbol,
)
from extraction.features.groundwater.utility import extract_date, extract_depth, extract_elevation
from extraction.features.stratigraphy.layer.layer import LayerDepthsEntry
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.data_extractor import (
    DataExtractor,
    FeatureOnPage,
)

logger = logging.getLogger(__name__)


MAX_DEPTH = 200  # Maximum depth of the groundwater in meters - Otherwise, depth might be confused with
# elevation from the extraction algorithm.
# TODO: One could use the depth column to find the maximal depth of the borehole and use this as a threshold.


class GroundwaterLevelExtractor(DataExtractor):
    """Extract groundwater information from a PDF document."""

    feature_name = "groundwater"

    # look for elevation values to the left, right and/or immediately below the key
    search_left_factor: float = 2
    search_right_factor: float = 8
    search_below_factor: float = 2
    search_above_factor: float = 2

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ü": "u"}

    def get_text_lines_near_key(self, groundwater_key_line: TextLine, lines: list[TextLine]) -> list[TextLine]:
        """Extracts the text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            groundwater_key_line (TextLine): the line containing the keyword that indicates a groundwater level
            lines (list[TextLine]): all the lines of text to search in

        Returns:
            list[TextLine]: all found lists of textlines that appeared around a key
        """
        key_rect = groundwater_key_line.rect
        groundwater_info_lines = self.get_lines_near_key(lines, groundwater_key_line)

        # sort the lines by their proximity to the key line center, compute the distance to the key line center
        key_center = (key_rect.x0 + key_rect.x1) / 2
        return sorted(groundwater_info_lines, key=lambda line: abs((line.rect.x0 + line.rect.x1) / 2 - key_center))

    def get_groundwater_from_lines(
        self, lines: list[TextLine], page_number: int, seen_depths: list[LayerDepthsEntry]
    ) -> FeatureOnPage[Groundwater] | None:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page_number (int): the page number (1-based) of the PDF document
            seen_depths (list[LayerDepthsEntry]): The list of already seen depths to avoid confusion with stratigraphy.

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
                if any(
                    depth_val == seen_depth.value and seen_depth.rect.intersects(line.rect)
                    for seen_depth in seen_depths
                ):
                    continue
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

        # return anyway, we can infer information later
        return FeatureOnPage(
            feature=Groundwater(depth=depth, date=date, elevation=elevation),
            rect=rect_union,
            page=page_number,
        )

    def extract_groundwater(
        self,
        page_number: int,
        text_lines: list[TextLine],
        geometric_lines: list[Line],
        extracted_boreholes: list[ExtractedBorehole],
    ) -> list[FeatureOnPage[Groundwater]]:
        """Extracts the groundwater information from a borehole profile.

        Args:
            page_number (int): The page number (1-indexed) of the PDF document.
            text_lines (list[TextLine]): The lines of text to extract the groundwater information from.
            geometric_lines (list[Line]): The geometric lines on the page.
            extracted_boreholes (list[ExtractedBorehole]): The extracted boreholes from the page.

        Returns:
            list[FeatureOnPage[Groundwater]]: the extracted coordinates (if any)
        """
        areas_of_interest: list[list[TextLine]] = []
        # extract text clues (e.g. GW)
        for groundwater_key_line in self.find_feature_key(text_lines):
            areas_of_interest.append(self.get_text_lines_near_key(groundwater_key_line, text_lines))
        # extract visual clues, like groundwater symbols
        for upper_symbol_geom_line in get_groundwater_symbol_upper_lines(text_lines, geometric_lines):
            areas_of_interest.append(get_text_lines_near_symbol(text_lines, upper_symbol_geom_line))
        # extract color clues: some documents highlight the reading in a distinct color
        for highlighted_line in get_minority_color_lines(text_lines):
            areas_of_interest.append(self.get_text_lines_near_key(highlighted_line, text_lines))

        seen_depths = [lay.depths for bh in extracted_boreholes for lay in bh.predictions if lay.depths]
        seen_depth_entries = [d for depth in seen_depths for d in (depth.start, depth.end) if d and d.rect]

        found_groundwaters = []
        for text_lines in areas_of_interest:
            found_groundwater = self.get_groundwater_from_lines(text_lines, page_number, seen_depth_entries)
            if found_groundwater:
                found_groundwaters.append(found_groundwater)

        if found_groundwaters:
            groundwater_output = ", ".join([str(entry.feature) for entry in found_groundwaters])
            logger.info("Found groundwater information on page %s: %s", page_number, groundwater_output)
            return found_groundwaters

        logger.info("No groundwater found in this borehole profile.")
        return []
