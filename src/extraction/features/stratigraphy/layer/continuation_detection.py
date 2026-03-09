"""This module contains functionality for detecting when a single borehole continues across pdf pages."""

import numpy as np

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer, LayerDepths, LayerDepthsEntry
from extraction.features.stratigraphy.layer.overlap_detection import select_boreholes_with_overlap
from swissgeol_doc_processing.text.textblock import MaterialDescription
from swissgeol_doc_processing.utils.file_utils import parse_text

DEPTHS_QUANTILE_SLACK = 0.1


def _reconcile_duplicated_boundary_layer(
    borehole_to_extend: ExtractedBorehole,
    borehole_continuation: ExtractedBorehole,
    upper_id: int,
    lower_id: int,
) -> ExtractedBorehole:
    """Extend the last kept layer of the upper borehole with the end depth of the first removed lower layer.

    This is used when duplicate layers are detected across overlapping pages. If the last kept upper layer and the
    first removed lower layer have exactly the same material description, the upper layer likely continues across the
    page break and should inherit the deeper end depth from the lower layer.

    Args:
        borehole_to_extend (ExtractedBorehole): The borehole from the previous page that is being extended.
        borehole_continuation (ExtractedBorehole): The borehole from the current page that is the continuation.
        upper_id (int): The index of the last layer in the upper borehole that is kept after removing duplicates.
        lower_id (int): The index of the first layer in the lower borehole that is removed as duplicate.

    Returns:
        ExtractedBorehole: The updated borehole.
    """
    if upper_id <= 0 or lower_id <= 0:
        return borehole_to_extend

    upper_layer = borehole_to_extend.predictions[upper_id - 1]
    lower_layer = borehole_continuation.predictions[lower_id - 1]

    if not _material_descriptions_match(upper_layer, lower_layer):
        return borehole_to_extend

    upper_depths = upper_layer.depths
    lower_depths = lower_layer.depths
    if not (
        upper_depths
        and upper_depths.start
        and upper_depths.end
        and lower_depths
        and lower_depths.end
        and upper_depths.start.value < lower_depths.end.value
    ):
        return borehole_to_extend

    reconciled_layer = Layer(
        material_description=MaterialDescription(
            text=upper_layer.material_description.text,
            lines=upper_layer.material_description.lines + lower_layer.material_description.lines,
        ),
        depths=LayerDepths(start=upper_depths.start, end=lower_depths.end),
    )

    updated_predictions = borehole_to_extend.predictions.copy()
    updated_predictions[upper_id - 1] = reconciled_layer

    return ExtractedBorehole(
        predictions=updated_predictions,
        bounding_boxes=borehole_to_extend.bounding_boxes,
    )


def _pick_merge_candidates(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
    page_number: int,
    matching_params: dict,
) -> tuple[ExtractedBorehole | None, list[ExtractedBorehole], ExtractedBorehole | None, list[ExtractedBorehole]]:
    """Select the most likely pair of boreholes to merge between consecutive pages.

    The method first attempts an overlap-based matching using scan alignment.
    If no overlap is detected, it falls back to depth/position-based matching.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): List of boreholes from the previous page.
        current_page_boreholes (list[ExtractedBorehole]): List of boreholes detected on the current page.
        page_number (int): Index of the current page being processed (1-based).
        matching_params (dict): The parameters for matching boreholes.

    Returns:
        tuple[ExtractedBorehole | None, list[ExtractedBorehole], ExtractedBorehole | None, list[ExtractedBorehole]]:
            * Extended borehole from the previous page.
            * Unaffected boreholes from the previous page.
            * Extended borehole from the current page that continues it.
            * Unaffected boreholes from the current page.
    """
    # 1) Overlap-based (returns a triple)
    borehole_to_extend, borehole_continuation, last_duplicated_layer_index = select_boreholes_with_overlap(
        previous_page_boreholes, current_page_boreholes, matching_params
    )

    if borehole_to_extend is None or borehole_continuation is None or last_duplicated_layer_index is None:
        # 2) Depth/position-based (returns a pair) → normalize to triple
        borehole_to_extend, borehole_continuation = _select_boreholes_for_concatenation(
            previous_page_boreholes, current_page_boreholes, page_number
        )

    unaffected_boreholes_previous_page = [
        borehole for borehole in previous_page_boreholes if borehole is not borehole_to_extend
    ]
    unaffected_boreholes_current_page = [
        borehole for borehole in current_page_boreholes if borehole is not borehole_continuation
    ]

    if last_duplicated_layer_index:
        upper_id, lower_id = last_duplicated_layer_index

        borehole_to_extend = _reconcile_duplicated_boundary_layer(
            borehole_to_extend, borehole_continuation, upper_id, lower_id
        )

        # Keep only the non-duplicated part of the borehole from the previous page
        borehole_to_extend = ExtractedBorehole(
            predictions=borehole_to_extend.predictions[:upper_id],
            bounding_boxes=borehole_to_extend.bounding_boxes,
        )

        # Keep only the new layers from the continuation borehole on the current
        borehole_continuation = ExtractedBorehole(
            predictions=borehole_continuation.predictions[lower_id:],
            bounding_boxes=borehole_continuation.bounding_boxes,
        )

    return (
        borehole_to_extend,
        unaffected_boreholes_previous_page,
        borehole_continuation,
        unaffected_boreholes_current_page,
    )


def merge_boreholes(
    boreholes_per_page: list[list[ExtractedBorehole]], matching_params: dict
) -> list[ExtractedBorehole]:
    """Merge boreholes that were extracted from different pages into continuous borehole.

    This method will attempt to match the predicted boreholes to the boreholes from previous pages based on their
    positions and continuity. If a match is found, the layers are merged; otherwise, new boreholes are added.

    Args:
        boreholes_per_page (list[list[ExtractedBorehole]]): List containing all boreholes with all layers per page.
        matching_params (dict): The parameters for matching boreholes.
    """
    finished_boreholes = []
    previous_page_boreholes = []

    for index, boreholes_on_page in enumerate(boreholes_per_page):
        page_number = index + 1

        # 1. merge boreholes that are continued from the previous page based on overlaps
        (
            borehole_to_extend,
            unaffected_boreholes_previous_page,
            borehole_continuation,
            unaffected_boreholes_current_page,
        ) = _pick_merge_candidates(previous_page_boreholes, boreholes_on_page, page_number, matching_params)

        # 2. If borehole merge possible, apply it
        if borehole_to_extend and borehole_continuation:
            merged_borehole = _merge_boreholes(borehole_to_extend, borehole_continuation)

            # Declare all unaffected boreholes from the previous page as finished
            finished_boreholes.extend(unaffected_boreholes_previous_page)

            # Put all boreholes from the current page in the list previous_page_boreholes for the next iteration
            previous_page_boreholes = unaffected_boreholes_current_page + [merged_borehole]
        else:
            # No merge possible, previous borehole considered as finished, current goes as previous
            finished_boreholes.extend(previous_page_boreholes)
            previous_page_boreholes = boreholes_on_page

    finished_boreholes.extend(previous_page_boreholes)
    for borehole in finished_boreholes:
        _normalize_first_layer(borehole)
    return finished_boreholes


def _select_boreholes_for_concatenation(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
    current_page_number: int,
) -> tuple[ExtractedBorehole | None, ExtractedBorehole | None]:
    """Select which boreholes (if any) on two adjacent pages should extend each other.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): The boreholes than can potentially be extended.
        current_page_boreholes (list[ExtractedBorehole]): The boreholes than can potentially be a continuation.
        current_page_number (int): The current page number.

    Returns:
        tuple[ExtractedBorehole | None, ExtractedBorehole | None]:
                A pair of boreholes, or None if nothing should be merged.
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


def _merge_boreholes(
    borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole
) -> ExtractedBorehole:
    """Merge the layers of the current borehole into the borehole to extend.

    If the last layer of the previous borehole and the first layer of the current borehole appear to be the same
    layer (based on depth continuity), they are merged into a single layer that spans both pages.

    Args:
        borehole_to_extend (ExtractedBorehole): The borehole from the previous page
        borehole_continuation (ExtractedBorehole): The borehole from the current page

    Returns:
        ExtractedBorehole: the merged borehole
    """
    # Do the last layer of the previous borehole and the first of the current belong to the same layer.
    last_prev_layer = borehole_to_extend.predictions[-1] if borehole_to_extend.predictions else None
    first_next_layer = borehole_continuation.predictions[0] if borehole_continuation.predictions else None

    new_predictions = borehole_to_extend.predictions + borehole_continuation.predictions

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

            spanning_layer = _build_spanning_layer(last_prev_layer, first_next_layer)
            new_predictions = (
                borehole_to_extend.predictions[:-1] + [spanning_layer] + borehole_continuation.predictions[1:]
            )
        elif (
            last_prev_depths
            and last_prev_depths.start
            and last_prev_depths.end
            and first_depths
            and first_depths.start
            and first_depths.end
            and _material_descriptions_match(last_prev_layer, first_next_layer)
            and last_prev_depths.start.value < first_depths.end.value
        ):
            # if the material description is repeated exactly across the page break, it likely indicates that
            # the layer continues on the next page and the end depth of the previous page corresponds only to
            # the page boundary, not the true end of the geological layer.

            spanning_layer = Layer(
                material_description=MaterialDescription(
                    text=last_prev_layer.material_description.text,
                    lines=last_prev_layer.material_description.lines + first_next_layer.material_description.lines,
                ),
                depths=LayerDepths(start=last_prev_depths.start, end=first_depths.end),
            )
            new_predictions = (
                borehole_to_extend.predictions[:-1] + [spanning_layer] + borehole_continuation.predictions[1:]
            )

    return ExtractedBorehole(
        predictions=new_predictions,
        bounding_boxes=borehole_to_extend.bounding_boxes + borehole_continuation.bounding_boxes,
    )


def _material_descriptions_match(previous_layer: Layer, current_layer: Layer) -> bool:
    """Check if two layers have exactly matching material descriptions after normalization.

    Args:
        previous_layer (Layer): The layer from the previous page.
        current_layer (Layer): The layer from the current page.

    Returns:
        bool: True if the material descriptions match, False otherwise.
    """
    previous_description = parse_text(previous_layer.material_description.text)
    current_description = parse_text(current_layer.material_description.text)
    return previous_description != "" and previous_description == current_description


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
