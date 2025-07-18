"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging

import cv2
import Levenshtein
import numpy as np
import pymupdf

from extraction.annotations.plot_utils import convert_page_to_opencv_img

from .layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


def remove_duplicate_layers(
    current_page_index: int,
    document: pymupdf.Document,
    previous_layers_with_bb: list[ExtractedBorehole],
    current_layers_with_bb: list[ExtractedBorehole],
    img_template_probability_threshold: float,
) -> list[ExtractedBorehole]:
    """Remove duplicate layers from the current page based on the layers of the previous page.

    We check if a layer on the current page is present on the previous page. If we have 3 consecutive layers that are
    not duplicates, we assume that there is no further overlap between the pages and stop the search. If we find a
    duplicate, all layers up to including the duplicate layer are removed.

    If the page contains a depth column, we compare the depth intervals and the material description to determine
    duplicate layers. If there is no depth column, we use template matching to compare the layers.

    Args:
        current_page_index: (int): the current page index (starting from 0)
        document (pymupdf.Document): The whole document.
        previous_layers_with_bb (list[ExtractedBorehole]): The layers of the previous page, with their bounding box.
        current_layers_with_bb (list[ExtractedBorehole]): The layers of the current page, with their bounding box.
        img_template_probability_threshold (float): The threshold for the template matching probability

    Returns:
        list[ExtractedBorehole]: The layers of the boreholes on the current page without duplicates. Their bounding
            boxes are kept the same.
    """
    if current_page_index == 0:
        return current_layers_with_bb

    previous_page = document[current_page_index - 1]
    current_page = document[current_page_index]

    non_duplicated_extracted_boreholes: list[ExtractedBorehole] = []
    # iterate on all the borehole profiles identified on this page
    for current_borehole_layers_with_bb in current_layers_with_bb:
        current_borehole_layers = current_borehole_layers_with_bb.predictions
        sorted_layers = sorted(current_borehole_layers, key=lambda x: x.material_description.rect.y0)
        first_non_duplicated_layer_index = 0
        count_consecutive_non_duplicate_layers = 0
        for layer_index, layer in enumerate(sorted_layers):
            # if we have 3 consecutive non-duplicate layers, we can assume that there is no further page overlap.
            if count_consecutive_non_duplicate_layers >= 3:
                break

            # check if current layer has an overlapping layer on the previous page.
            # for that purpose compare depth interval as well as material description text.
            duplicate_condition = False
            if layer.depths is None:
                duplicate_condition = check_duplicate_layer_by_template_matching(
                    previous_page, current_page, layer, img_template_probability_threshold
                )
            else:  # in this case we compare the depth interval and material description
                current_material_description = layer.material_description
                current_depth_interval = layer.depths
                # iterate on all the layers in the previously identified borehole profiles
                for previous_borehole_layers_with_bb in previous_layers_with_bb:
                    previous_borehole_layers = previous_borehole_layers_with_bb.predictions
                    for previous_layer in previous_borehole_layers:
                        if previous_layer.depths is None:
                            # It may happen, that a layer on the previous page does not have depth interval assigned.
                            # In this case we skip the comparison. This should only happen in some edge cases, as we
                            # assume that when the current page has a depth column, that the previous page also
                            # contains a depth column.
                            # We assume overlapping pages and a depth column should extend over both pages.
                            continue

                        previous_material_description = previous_layer.material_description
                        previous_depth_interval = previous_layer.depths

                        # start values for the depth intervals may be None. End values are always explicitly set.
                        current_depth_interval_start = (
                            current_depth_interval.start.value
                            if ((current_depth_interval is not None) and (current_depth_interval.start is not None))
                            else None
                        )
                        previous_depth_interval_start = (
                            previous_depth_interval.start.value
                            if ((previous_depth_interval is not None) and (previous_depth_interval.start is not None))
                            else None
                        )
                        # check if material description is the same
                        text_similarity = (
                            Levenshtein.ratio(
                                current_material_description.feature.text, previous_material_description.feature.text
                            )
                            > 0.9
                        )

                        same_start_depth = current_depth_interval_start == previous_depth_interval_start
                        if current_depth_interval.end and previous_depth_interval.end:
                            same_end_depth = current_depth_interval.end.value == previous_depth_interval.end.value

                            if text_similarity and same_start_depth and same_end_depth:
                                duplicate_condition = True
                                logger.info("Removing duplicate layer.")
                                break

                    if duplicate_condition:
                        break
            # the layers of the current borehole are duplicates of the layers of one of the previous borehole
            if duplicate_condition:
                first_non_duplicated_layer_index = layer_index + 1  # all layers before this layer are duplicates
                count_consecutive_non_duplicate_layers = 0
            else:
                count_consecutive_non_duplicate_layers += 1
        non_duplicated_extracted_boreholes.append(
            ExtractedBorehole(
                predictions=sorted_layers[first_non_duplicated_layer_index:],
                bounding_boxes=current_borehole_layers_with_bb.bounding_boxes,
            )
        )

    return non_duplicated_extracted_boreholes


def check_duplicate_layer_by_template_matching(
    previous_page: pymupdf.Page,
    current_page: pymupdf.Page,
    current_layer: Layer,
    img_template_probability_threshold: float,
) -> bool:
    """Check if the current layer is a duplicate of a layer on the previous page by using template matching.

    This is done by extracting an image of the layer and check if that image is present in the previous page
    by applying template matching onto the previous page. This checks if the image of the current layer is present
    in the previous page.

    Args:
        previous_page (pymupdf.Page): The previous page.
        current_page (pymupdf.Page): The current page.
        current_layer (Layer): The current layer that is checked for a duplicate.
        img_template_probability_threshold (float): The threshold for the template matching probability
                                                    to consider a layer a duplicate.

    Returns:
        bool: True if the layer is a duplicate, False otherwise.
    """
    scale_factor = 3
    current_page_image = convert_page_to_opencv_img(
        current_page, scale_factor=scale_factor, color_mode=cv2.COLOR_BGR2GRAY
    )
    previous_page_image = convert_page_to_opencv_img(
        previous_page, scale_factor=scale_factor, color_mode=cv2.COLOR_BGR2GRAY
    )

    [x0, y_start, x1, y_end] = current_layer.material_description.rect
    x_start = int(scale_factor * min(x0, current_page.rect.width * 0.2))  # 0.2 is a magic number that works well
    x_end = int(scale_factor * min(max(x1, current_page.rect.width * 0.8), previous_page.rect.width - 1))
    y_start = int(scale_factor * max(y_start, 0))  # do not go higher up as otherwise we remove too many layers.
    y_end = int(scale_factor * min(y_end + 5, previous_page.rect.height - 1, current_page.rect.height - 1))
    # y_start and y_end define the upper and lower bound of the image used to compare to the previous page
    # and determine if there is an overlap. We add 5 pixel to y_end to add a bit more context to the image
    # as the material_description bounding box is very tight around the text. Furthermore, we need to ensure
    # that the template is smaller than the previous and the current page.
    # y_start should not be lowered further as otherwise the we include potential overlap to the previous page
    # that belongs to the previous layer.

    layer_image = current_page_image[y_start:y_end, x_start:x_end]
    try:
        img_template_probablility_match = np.max(
            cv2.matchTemplate(previous_page_image, layer_image, cv2.TM_CCOEFF_NORMED)
        )
    except cv2.error:  # there can be strange correlation errors here.
        # Just ignore them as it is only a few over the complete dataset
        logger.warning("Error in template matching. Skipping layer.")
        return False
    return img_template_probablility_match > img_template_probability_threshold
