"""This module contains functionality for detecting when a single borehole continues across pdf pages."""

import numpy as np

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer, LayerDepths, LayerDepthsEntry
from extraction.features.stratigraphy.layer.overlap_detection import select_boreholes_with_scan_overlap
from extraction.features.utils.text.textblock import MaterialDescription

DEPTHS_QUANTILE_SLACK = 0.1


def merge_boreholes(boreholes_per_page: list[list[ExtractedBorehole]]) -> list[ExtractedBorehole]:
    """Merge boreholes that were extracted from different pages.

    This method will attempt to match the predicted boreholes to the boreholes from previous pages based on their
    positions and continuity. If a match is found, the layers are merged; otherwise, new boreholes are added.

    Args:
        boreholes_per_page (list[list[ExtractedBorehole]]): List containing all boreholes with all layers per page.
    """
    finished_boreholes = []
    previous_page_boreholes = []

    for index, boreholes_on_page in enumerate(boreholes_per_page):
        page_number = index + 1

        # 1. merge boreholes that are continued from the previous page based on overlaps
        borehole_to_extend, borehole_continuation, duplicate_layer_count = select_boreholes_with_scan_overlap(
            previous_page_boreholes=previous_page_boreholes,
            current_page_boreholes=boreholes_on_page,
        )

        # 2. alternatively, merge boreholes that are continued from the previous page based on depths and position
        if not (borehole_to_extend and borehole_continuation and duplicate_layer_count):
            borehole_to_extend, borehole_continuation = _select_boreholes_for_merge(
                previous_page_boreholes, boreholes_on_page, page_number
            )

        if borehole_to_extend and borehole_continuation:
            other_boreholes_previous = [
                borehole for borehole in previous_page_boreholes if borehole is not borehole_to_extend
            ]
            other_boreholes_current = [
                borehole for borehole in boreholes_on_page if borehole is not borehole_continuation
            ]
            if duplicate_layer_count is not None:
                borehole_continuation = ExtractedBorehole(
                    predictions=borehole_continuation.predictions[duplicate_layer_count + 1 :],
                    bounding_boxes=borehole_continuation.bounding_boxes,
                )
            _merge_boreholes(borehole_to_extend, borehole_continuation)

            finished_boreholes.extend(other_boreholes_previous)
            previous_page_boreholes = other_boreholes_current
            previous_page_boreholes.append(borehole_to_extend)
        else:
            finished_boreholes.extend(previous_page_boreholes)
            previous_page_boreholes = boreholes_on_page

    finished_boreholes.extend(previous_page_boreholes)
    for borehole in finished_boreholes:
        _normalize_first_layer(borehole)
    return finished_boreholes


def _select_boreholes_for_merge(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
    current_page_number: int,
) -> (ExtractedBorehole | None, ExtractedBorehole | None):
    """Select which boreholes (if any) on two adjacent pages should extend each other.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): The boreholes than can potentially be extended.
        current_page_boreholes (list[ExtractedBorehole]): The boreholes than can potentially be a continuation.
        current_page_number (int): The current page number.

    Returns:
        (ExtractedBorehole, ExtractedBorehole) | None: a pair of boreholes, or None if nothing should be merged.
    """
    if previous_page_boreholes and current_page_boreholes:
        # 1. Out of the previously detected boreholes, determine which one should be continued.
        borehole_to_extend = _identify_borehole_to_extend(previous_page_boreholes, current_page_number)

        # 2. If there is more than one borehole on the current page, determine which one should extend the previous
        borehole_continuation = _identify_borehole_continuation(current_page_boreholes)

        # 3. Is the current borehole likely to be the continuation of the previous.
        if _is_continuation(borehole_to_extend, borehole_continuation, current_page_number):
            return borehole_to_extend, borehole_continuation

    return None, None


def _identify_borehole_to_extend(boreholes: list[ExtractedBorehole], current_page: int) -> ExtractedBorehole | None:
    """Identify the borehole to potentially extend with the layers on this page.

    If there is more than one previous borehole, we rank them to determine the most likely to be continued. The
    score uses y1 (lower means more likely to continue on the next page) and also the width, so that
    narrow parasite boreholes don't get too much weight.

    Args:
        boreholes (list[ExtractedBorehole]): The boreholes than can potentially be extended.
        current_page (int): The current page number.

    Returns:
        ExtractedBorehole | None: The borehole to extend, or None if no borehole is found on the previous page.
    """
    assert current_page > 1, "Can't be here on the first page"

    def get_score(current_page, borehole):
        prev_page_bbox = next((bbox for bbox in borehole.bounding_boxes if bbox.page == current_page - 1), None)
        if prev_page_bbox is None:
            return float("-inf")
        outer_rect = prev_page_bbox.get_outer_rect()
        return outer_rect.y1 + outer_rect.width

    return max(boreholes, key=lambda borehole: get_score(current_page, borehole))


def _identify_borehole_continuation(boreholes: list[ExtractedBorehole]) -> ExtractedBorehole:
    """Identify the borehole on the current page that is most likely to be the continuation of the previous.

    Args:
        boreholes (list[ExtractedBorehole]): List containing all detected boreholes on this page.

    Returns:
        ExtractedBorehole: The borehole that is most likely to be the continuation of the previous.
    """
    return min(boreholes, key=lambda bh: bh.bounding_boxes[0].get_outer_rect().y0)


def _is_continuation(
    borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole, current_page: int
) -> bool:
    """Determine if the borehole on the current page is the continuation of the previous borehole.

    This method check if the depths are continuous, if the sidebar bounding boxes overlap or if the material
    description bounding boxes significantly.

    Args:
        borehole_to_extend (ExtractedBorehole): The borehole from the previous page
        borehole_continuation (ExtractedBorehole): The borehole from the current page
        current_page (int): The current page number

    Returns:
        bool: True if the current borehole is the continuation of the previous borehole, False otherwise.
    """
    ok_prev_layers = [lay for lay in borehole_to_extend.predictions if lay.depths is not None]
    prev_depths = [d.value for lay in ok_prev_layers for d in (lay.depths.start, lay.depths.end) if d is not None]

    ok_layers = [lay for lay in borehole_continuation.predictions if lay.depths is not None]
    depths = [d.value for lay in ok_layers for d in (lay.depths.start, lay.depths.end) if d is not None]
    # use quantile to allow some slack (e.g. few undetected duplicated layers)
    # if depth values decrease, it must be a new borehole
    return not (depths and prev_depths and depths[0] < np.quantile(prev_depths, 1 - DEPTHS_QUANTILE_SLACK))


def _merge_boreholes(borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole):
    """Merge the layers of the current borehole into the borehole to extend.

    If the last layer of the previous borehole and the first layer of the current borehole appear to be the same
    layer (based on depth continuity), they are merged into a single layer that spans both pages.

    Args:
        borehole_to_extend (ExtractedBorehole): The borehole from the previous page
        borehole_continuation (ExtractedBorehole): The borehole from the current page

    Note:
        This method modifies the state of the borehole_to_extend object by updating its predictions and bounding
    """
    # Do the last layer of the previous borehole and the first of the current belong to the same layer.
    last_prev_layer = borehole_to_extend.predictions[-1] if borehole_to_extend.predictions else None
    first_next_layer = borehole_continuation.predictions[0] if borehole_continuation.predictions else None

    if last_prev_layer and first_next_layer:
        last_prev_depths = last_prev_layer.depths
        first_depths = first_next_layer.depths
        if (
            (last_prev_depths and last_prev_depths.start and last_prev_depths.end is None)
            and (first_depths and first_depths.start is None and first_depths.end)
            and (last_prev_depths.start.value < first_depths.end.value)
        ):
            # if the last interval of the previous page is open-ended, and the first of this page has no start
            # value, it probably means that they refer to the same layer.

            borehole_to_extend.predictions[-1] = _build_spanning_layer(last_prev_layer, first_next_layer)
            borehole_continuation.predictions = borehole_continuation.predictions[1:]
    borehole_to_extend.bounding_boxes.extend(borehole_continuation.bounding_boxes)
    borehole_to_extend.predictions.extend(borehole_continuation.predictions)


def _build_spanning_layer(last_prev_layer: Layer, first_next_layer: Layer) -> Layer:
    """Build a new layer that spans the last layer of the previous borehole and the first of the current.

    Args:
        last_prev_layer (Layer): The first layer of the previous borehole.
        first_next_layer (Layer): The first layer of the current borehole.

    Returns:
        Layer: A new layer that contains the material description of both layers and spans their depths.
    """
    return Layer(
        material_description=MaterialDescription(
            text=last_prev_layer.material_description.text + " " + first_next_layer.material_description.text,
            lines=last_prev_layer.material_description.lines + first_next_layer.material_description.lines,
        ),
        depths=LayerDepths(start=last_prev_layer.depths.start, end=first_next_layer.depths.end),
    )


def _normalize_first_layer(borehole: ExtractedBorehole):
    """Normalize the first layer of a borehole.

    This means that for the first layer of each borehole, if it has no start depth, but has an end depth, we set
    the start depth to 0. This is due to the depth 0.0 often being not explicitly mentioned in the text.
    This step is done after all layers have been assigned to boreholes, so that we can be sure that we have the
    full context of the borehole.
    """
    if (
        borehole.predictions
        and borehole.predictions[0].depths
        and borehole.predictions[0].depths.start is None
        and borehole.predictions[0].depths.end is not None
    ):
        end_page = borehole.predictions[0].depths.end.page_number
        borehole.predictions[0].depths.start = LayerDepthsEntry(0.0, None, end_page)
