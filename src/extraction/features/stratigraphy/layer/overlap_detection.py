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
    """_summary_.

    Args:
        layer_prev (Layer): _description_
        layer_curr (Layer): _description_
        material_threshold (float): _description_
        force_depth_matching (bool, optional): _description_. Defaults to False.
        is_extremity (bool, optional): _description_. Defaults to False.

    Returns:
        bool: _description_
    """
    # Check how close two layers are from each other based on depth and material description
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
        if layer_curr.depths.start and layer_prev.depths.start and layer_curr.depths.start != layer_prev.depths.start:
            return False
        # Rule 3.2 Ending depth should match
        if layer_curr.depths.end and layer_prev.depths.end and layer_curr.depths.end != layer_prev.depths.end:
            return False

    # All rules validated
    return True


def find_split_by_convolution(layers_prev: list[Layer], layers_curr: list[Layer], matching_params: dict) -> int | None:
    """_summary_. Find first layer in current that does not overlap with previous. If none, no overlap.

    Args:
        layers_prev (list[Layer]): _description_
        layers_curr (list[Layer]): _description_
        matching_params (dict): _description_

    Returns:
        int | None: _description_
    """
    material_threshold = matching_params["duplicate_layer_threshold"]
    force_depth = matching_params["duplicate_layer_force_depth"]
    idx_best = None

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
            idx_best = i

    return idx_best


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
        is_extremity (bool): TODO

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
