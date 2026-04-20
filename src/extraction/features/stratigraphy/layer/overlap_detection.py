"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging

import Levenshtein

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


def select_boreholes_with_overlap(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
    matching_params: dict,
) -> tuple[ExtractedBorehole | None, ExtractedBorehole | None, tuple[int, int] | None]:
    """Remove duplicate layers caused by overlapping scanned pages.

    Compare layers from current page with those from previous page to identify and remove
    duplicates. Layers are compared from bottom to top based on their material descriptions.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): Layers from previous page
        current_page_boreholes (list[ExtractedBorehole]): Layers from current page
        matching_params (dict): The parameters for matching boreholes.

    Returns:
        (ExtractedBorehole | None, ExtractedBorehole | None, tuple[int, int] | None):
                The boreholes to be extended, the continuing borehole, and the indexes of the first and last
                duplicated layer in the previous and continuing borehole, if any.
    """
    for current_borehole in current_page_boreholes:
        for previous_page_borehole in previous_page_boreholes:
            # # Check overlap between layers
            if id_best := find_split_by_convolution(
                previous_page_borehole.predictions, current_borehole.predictions, matching_params
            ):
                upper_id = len(previous_page_borehole.predictions)
                lower_id = id_best

                return previous_page_borehole, current_borehole, (upper_id, lower_id)

    return None, None, None


def _are_layers_similar(
    layer_prev: Layer,
    layer_curr: Layer,
    material_threshold: float,
    force_depth_matching: bool = False,
    is_extremity: bool = False,
) -> bool:
    """Check if two layers are similar based on material description and optional depth matching.

    Validates layers through three rules:
    1. Both layers must have material descriptions
    2. Material descriptions must exceed similarity threshold
    3. If force_depth_matching is enabled, depths must match

    Args:
        layer_prev (Layer): The layer from the previous page.
        layer_curr (Layer): The layer from the current page.
        material_threshold (float): Minimum similarity threshold for material descriptions.
        force_depth_matching (bool, optional): Whether to enforce matching depths. Defaults to False.
        is_extremity (bool, optional): Whether this layer is at the boundary of the overlap. Defaults to False.

    Returns:
        bool: True if layers are similar according to all active rules, False otherwise.
    """
    # Rule 1: Material description should exists
    if not layer_prev.material_description or not layer_curr.material_description:
        return False

    # Rule 2: Material description score should exceed threshold
    if not _is_duplicate(
        cur_text=layer_curr.material_description.text,
        prev_text=layer_prev.material_description.text,
        t=material_threshold,
        is_extremity=is_extremity,
    ):
        return False

    # Rule 3: If depth exists, should match
    if force_depth_matching and layer_curr.depths and layer_prev.depths:
        # Rule 3.1 Strating depth should match
        if (
            layer_curr.depths.start
            and layer_prev.depths.start
            and layer_curr.depths.start.value != layer_prev.depths.start.value
        ):
            return False
        # Rule 3.2 Ending depth should match
        if (
            layer_curr.depths.end
            and layer_prev.depths.end
            and layer_curr.depths.end.value != layer_prev.depths.end.value
        ):
            return False

    # All rules validated
    return True


def find_split_by_convolution(layers_prev: list[Layer], layers_curr: list[Layer], matching_params: dict) -> int | None:
    """Find the extent of overlap between consecutive page layers.

    Determines the maximum number of consecutive layers from the bottom of the previous
    page that match the top layers of the current page.

    Args:
        layers_prev (list[Layer]): Layers from the previous page, ordered top to bottom.
        layers_curr (list[Layer]): Layers from the current page, ordered top to bottom.
        matching_params (dict): Configuration dict with keys.

    Returns:
        int | None: Index of first non-overlapping layer in current page, or None if no overlap.
    """
    material_threshold = matching_params["duplicate_layer_threshold"]
    force_depth = matching_params["duplicate_layer_force_depth"]
    idx_next = None

    for i in list(range(1, min(len(layers_prev), len(layers_curr)) + 1)):
        # All layers should match to enforce overlap
        if all(
            _are_layers_similar(
                layer_prev=layer_prev,
                layer_curr=layer_curr,
                material_threshold=material_threshold,
                force_depth_matching=force_depth,
                is_extremity=(j == 0 or j == i - 1),  # Indicate to function that one layer might be cut (extremities)
            )
            for j, (layer_prev, layer_curr) in enumerate(zip(layers_prev[-i:], layers_curr[:i], strict=True))
        ):
            idx_next = i

    return idx_next


def _is_duplicate(cur_text: str, prev_text: str, t: float, is_extremity: bool):
    """Detect if layer and prev_layer are duplicates across a page break.

    Strategy:
      - Check any suffix of prev_text words vs the full current text.
      - Check any prefix of current words vs the full prev_text text.
    This covers both split cases (the prev_text was cut off / the cur_text only repeats the tail).

    Args:
        cur_text (str): The text of the current layer to compare.
        prev_text (str): The text of the previous layer to compare against.
        t (float): The similarity threshold.
        is_extremity (bool): If True, uses partial text matching to handle truncated boundary layers.
            If False, compares full text content.

    Returns:
        bool: True if the layers are considered duplicates, False otherwise.
    """
    cur_text = cur_text.lower()
    prev_text = prev_text.lower()

    if is_extremity:
        min_length = min(len(cur_text), len(prev_text))
        score = max(
            Levenshtein.ratio(cur_text, prev_text[-min_length:]),
            Levenshtein.ratio(cur_text[:min_length], prev_text),
        )
    else:
        score = Levenshtein.ratio(cur_text, prev_text)

    return score > t
