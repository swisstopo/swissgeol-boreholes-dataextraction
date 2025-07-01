"""This module contains the template matching code.

The code in this file aims to extract groundwater information based on the location where
the groundwater illustration was found in the document of interest.
"""

import logging
import math
import os
from pathlib import Path

import numpy as np
import pymupdf
import skimage as ski

from extraction.features.groundwater.groundwater_extraction import Groundwater, GroundwaterLevelExtractor
from extraction.features.utils.data_extractor import FeatureOnPage
from extraction.features.utils.text.textline import TextLine
from extraction.features.utils.utility import get_lines_near_rect

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
    document: pymupdf.Document,
) -> tuple[list[FeatureOnPage[Groundwater]], list[float]]:
    """Extracts the groundwater information from an illustration.

    Args:
        groundwater_extractor (GroundwaterLevelExtractor): the groundwater level extractor.
        lines (list[TextLine]): The lines of text to extract the groundwater information from.
        page_number (int): The page number (1-based) of the PDF document.
        document (pymupdf.Document): The document to extract groundwater from illustration from.

    Returns:
        list[FeatureOnPage[Groundwater]]: the extracted groundwater information
        list[float]: the confidence of the extraction
    """
    extracted_groundwater_list = []
    confidence_list = []

    # convert the doc to an image
    page = document.load_page(page_number - 1)
    filename = Path(document.name).stem
    png_filename = f"{filename}-{page_number + 1}.png"
    png_path = f"/tmp/{png_filename}"  # Local path to save the PNG
    pymupdf.utils.get_pixmap(page, matrix=pymupdf.Matrix(2, 2), clip=page.rect).save(png_path)

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
            illustration_rect = pymupdf.Rect(
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
            pdf_illustration_rect = pymupdf.Rect(
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
            def distance_to_key_center(line_rect: pymupdf.Rect, illustration_rect: pymupdf.Rect) -> float:
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
                if (
                    (extracted_gw.groundwater.depth or extracted_gw.groundwater.elevation)
                    and extracted_gw not in extracted_groundwater_list
                    and extracted_gw.groundwater.date
                ):
                    # Only if the groundwater information is not already in the list
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
