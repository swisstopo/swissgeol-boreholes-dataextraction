"""This module contains the template matching code.

The code in this file aims to extract groundwater information based on the location where
the groundwater illustration was found in the document of interest.
"""

import logging
import math
import os
from pathlib import Path

import fitz
import numpy as np
import skimage as ski
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.data_extractor.utility import get_lines_near_rect
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterLevelExtractor
from stratigraphy.lines.line import TextLine
from stratigraphy.metadata.elevation_extraction import Elevation

logger = logging.getLogger(__name__)


def load_templates() -> list[np.ndarray]:
    """Load the templates for the groundwater information.

    Returns:
        list[np.ndarray]: the loaded templates
    """
    templates = []
    template_dir = os.path.join(os.path.dirname(__file__), "assets")
    for template in os.listdir(template_dir):
        if template.endswith(".npy"):
            templates.append(np.load(os.path.join(template_dir, template)))
    return templates


def get_groundwater_from_illustration(
    groundwater_extractor: GroundwaterLevelExtractor,
    lines: list[TextLine],
    page_number: int,
    terrain_elevation: Elevation | None,
) -> tuple[list[FeatureOnPage[Groundwater]], list[float]]:
    """Extracts the groundwater information from an illustration.

    Args:
        groundwater_extractor (GroundwaterLevelExtractor): the groundwater level extractor
        lines (list[TextLine]): the lines of text to extract the groundwater information from
        page_number (int): the page number (1-based) of the PDF document
        terrain_elevation (Elevation | None): The elevation of the terrain.

    Returns:
        list[FeatureOnPage[Groundwater]]: the extracted groundwater information
        list[float]: the confidence of the extraction
    """
    extracted_groundwater_list = []
    confidence_list = []

    # convert the doc to an image
    page = groundwater_extractor.doc.load_page(page_number - 1)
    filename = Path(groundwater_extractor.doc.name).stem
    png_filename = f"{filename}-{page_number + 1}.png"
    png_path = f"/tmp/{png_filename}"  # Local path to save the PNG
    fitz.utils.get_pixmap(page, matrix=fitz.Matrix(2, 2), clip=page.rect).save(png_path)

    # load the image
    img = ski.io.imread(png_path)
    N_BEST_MATCHES = 5
    TEMPLATE_MATCH_THRESHOLD = 0.66

    # extract the groundwater information from the image
    for template in load_templates():
        # Compute the match of the template and the image (correlation coef)
        result = ski.feature.match_template(img, template)

        for _ in range(N_BEST_MATCHES):
            ij = np.unravel_index(np.argmax(result), result.shape)
            confidence = np.max(result)  # TODO - use confidence to filter out bad matches
            if confidence < TEMPLATE_MATCH_THRESHOLD:
                # skip this template if the confidence is too low to avoid false positives
                continue
            top_left = (ij[1], ij[0])
            illustration_rect = fitz.Rect(
                top_left[0], top_left[1], top_left[0] + template.shape[1], top_left[1] + template.shape[0]
            )

            # Remove the matched area from the template matching result to avoid finding the same area again
            # for the same template
            x_area_to_remove = int(0.75 * template.shape[1])
            y_area_to_remove = int(0.75 * template.shape[0])
            result[
                int(illustration_rect.y0) - y_area_to_remove : int(illustration_rect.y1) + y_area_to_remove,
                int(illustration_rect.x0) - x_area_to_remove : int(illustration_rect.x1) + x_area_to_remove,
            ] = float("-inf")

            # convert the illustration_rect to the coordinate system of the PDF
            horizontal_scaling = page.rect.width / img.shape[1]
            vertical_scaling = page.rect.height / img.shape[0]
            pdf_illustration_rect = fitz.Rect(
                illustration_rect.x0 * horizontal_scaling,
                illustration_rect.y0 * vertical_scaling,
                illustration_rect.x1 * horizontal_scaling,
                illustration_rect.y1 * vertical_scaling,
            )

            # extract the groundwater information from the image using the text
            groundwater_info_lines = get_lines_near_rect(
                groundwater_extractor.search_left_factor,
                groundwater_extractor.search_right_factor,
                groundwater_extractor.search_above_factor,
                groundwater_extractor.search_below_factor,
                lines,
                pdf_illustration_rect,
            )

            # sort the lines by their proximity to the key line center, compute the distance to the key line center
            def distance_to_key_center(line_rect: fitz.Rect, illustration_rect: fitz.Rect) -> float:
                key_center_x = (illustration_rect.x0 + illustration_rect.x1) / 2
                key_center_y = (illustration_rect.y0 + illustration_rect.y1) / 2
                line_center_x = (line_rect.x0 + line_rect.x1) / 2
                line_center_y = (line_rect.y0 + line_rect.y1) / 2
                return math.sqrt((line_center_x - key_center_x) ** 2 + (line_center_y - key_center_y) ** 2)

            groundwater_info_lines.sort(key=lambda line: distance_to_key_center(line.rect, pdf_illustration_rect))
            try:
                extracted_gw = groundwater_extractor.get_groundwater_info_from_lines(
                    groundwater_info_lines, page_number
                )
                if extracted_gw.groundwater.depth or extracted_gw.groundwater.elevation:
                    # Fill in the depth and elevation if they are not already filled in based on the terrain
                    if terrain_elevation:
                        if not extracted_gw.groundwater.depth and extracted_gw.groundwater.elevation:
                            extracted_gw.groundwater.depth = round(
                                terrain_elevation.elevation - extracted_gw.groundwater.elevation, 2
                            )
                        if not extracted_gw.groundwater.elevation and extracted_gw.groundwater.depth:
                            extracted_gw.groundwater.elevation = round(
                                terrain_elevation.elevation - extracted_gw.groundwater.depth, 2
                            )

                        # Make a sanity check to see if elevation and depth make sense (i.e., they add up:
                        # elevation + depth = terrain elevation)
                        if extracted_gw.groundwater.elevation and extracted_gw.groundwater.depth:
                            extract_terrain_elevation = round(
                                extracted_gw.groundwater.elevation + extracted_gw.groundwater.depth, 2
                            )
                            if extract_terrain_elevation != terrain_elevation.elevation:
                                # If the extracted elevation and depth do not match the terrain elevation, we try
                                # to remove one of the items from the match and see if we can find a better match.
                                logger.warning("The extracted elevation and depth do not match the terrain elevation.")
                                logger.warning(
                                    "Elevation: %s, Depth: %s, Terrain Elevation: %s",
                                    extracted_gw.groundwater.elevation,
                                    extracted_gw.groundwater.depth,
                                    terrain_elevation.elevation,
                                )

                                # re-run the extraction and see if we can find a better match by removing one
                                # item from the current match
                                groundwater_info_lines_without_depth = [
                                    line
                                    for line in groundwater_info_lines
                                    if str(extracted_gw.groundwater.depth) not in line.text
                                ]
                                groundwater_info_lines_without_elevation = [
                                    line
                                    for line in groundwater_info_lines
                                    if str(extracted_gw.groundwater.elevation) not in line.text
                                ]
                                extracted_gw = groundwater_extractor.get_groundwater_info_from_lines(
                                    groundwater_info_lines_without_depth, page_number
                                )

                                if not extracted_gw.groundwater.depth:
                                    extracted_gw = groundwater_extractor.get_groundwater_info_from_lines(
                                        groundwater_info_lines_without_elevation, page_number
                                    )

                                if extracted_gw.groundwater.elevation and extracted_gw.groundwater.depth:
                                    extract_terrain_elevation = round(
                                        extracted_gw.groundwater.elevation + extracted_gw.groundwater.depth, 2
                                    )

                                    if extract_terrain_elevation != terrain_elevation.elevation:
                                        logger.warning(
                                            "The extracted elevation and depth do not match the terrain elevation."
                                        )
                                        logger.warning(
                                            "Elevation: %s, Depth: %s, Terrain Elevation: %s",
                                            extracted_gw.groundwater.elevation,
                                            extracted_gw.groundwater.depth,
                                            terrain_elevation.elevation,
                                        )
                                        continue

                    # Only if the groundwater information is not already in the list
                    if extracted_gw not in extracted_groundwater_list and extracted_gw.groundwater.date:
                        extracted_groundwater_list.append(extracted_gw)
                        confidence_list.append(confidence)

                        # Remove the extracted groundwater information from the lines to avoid double extraction
                        for line in groundwater_info_lines:
                            # if the rectangle of the line is in contact with the rectangle of the extracted
                            # groundwater information, remove the line
                            if line.rect.intersects(extracted_gw.rect):
                                lines.remove(line)

            except ValueError as error:
                logger.warning("ValueError: %s", error)
                continue

        # TODO: Maybe we could stop the search if we found a good match with one of the templates

    return extracted_groundwater_list, confidence_list
