"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging
import math
from dataclasses import dataclass

import Levenshtein

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


@dataclass
class OverlapResult:
    """Data class that contains all information regarding overlapping layers on consecutive pages.

    `upper_id` is the number of layers to keep from the previous borehole and `lower_id` is the index of the
    first non-overlapping layer in the continuing borehole.
    """

    upper_id: int
    lower_id: int


def select_boreholes_with_overlap(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
    matching_params: dict,
) -> tuple[ExtractedBorehole | None, ExtractedBorehole | None, OverlapResult | None]:
    """Remove duplicate layers caused by overlapping scanned pages.

    Compare layers from current page with those from previous page using a sliding-window
    approach to find the longest contiguous overlap at the page boundary.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): Layers from previous page
        current_page_boreholes (list[ExtractedBorehole]): Layers from current page
        matching_params (dict): The parameters for matching boreholes.

    Returns:
        (ExtractedBorehole | None, ExtractedBorehole | None, OverlapResult | None):
            The borehole to be extended, the continuing borehole, and the precise location of the overlap.
    """
    for current_borehole in current_page_boreholes:
        for previous_page_borehole in previous_page_boreholes:
            # Check overlap between layers
            if overlap := find_split_by_convolution(
                previous_page_borehole.predictions, current_borehole.predictions, matching_params
            ):
                return previous_page_borehole, current_borehole, overlap

    return None, None, None


def are_layers_similar(
    layer_prev: Layer,
    layer_curr: Layer,
    material_threshold: float = 0.95,
    is_extremity: bool = False,
) -> bool:
    """Check if two layers are similar based on material description and optional depth matching.

    Validates layers through three rules:
    1. Both layers must have material descriptions
    2. Material descriptions must exceed similarity threshold
    3. It present, depths must match

    Args:
        layer_prev (Layer): The layer from the previous page.
        layer_curr (Layer): The layer from the current page.
        material_threshold (float): Minimum similarity threshold for material descriptions. Defaults to 0.95.
        is_extremity (bool, optional): Whether this layer is at the boundary of the overlap. Defaults to False.

    Returns:
        bool: True if layers are similar according to all active rules, False otherwise.
    """
    # Rule 1: Material descriptions must exist
    if not layer_prev.material_description or not layer_curr.material_description:
        return False

    # Rule 2: Material description score should exceed threshold
    if not _is_duplicate(
        cur_text=layer_curr.material_description.text,
        prev_text=layer_prev.material_description.text,
        threshold=material_threshold,
        is_extremity=is_extremity,
    ):
        return False

    # Rule 3: If depth exists, should match
    if layer_curr.depths and layer_prev.depths:
        # Rule 3.1 Starting depth should match within tolerance
        if (
            layer_curr.depths.start
            and layer_curr.depths.start.value
            and layer_prev.depths.start
            and layer_prev.depths.start.value
            and not math.isclose(layer_curr.depths.start.value, layer_prev.depths.start.value, abs_tol=1e-2)
        ):
            return False
        # Rule 3.2 Ending depth should match within tolerance
        if (
            layer_curr.depths.end
            and layer_curr.depths.end.value
            and layer_prev.depths.end
            and layer_prev.depths.end.value
            and not math.isclose(layer_curr.depths.end.value, layer_prev.depths.end.value, abs_tol=1e-2)
        ):
            return False

    # All rules validated
    return True


def find_split_by_convolution(
    layers_prev: list[Layer], layers_curr: list[Layer], matching_params: dict
) -> OverlapResult | None:
    """Find the extent of overlap between consecutive page layers.

    Determines the maximum number of consecutive layers from the bottom of the previous
    page that match the top layers of the current page.

    Args:
        layers_prev (list[Layer]): Layers from the previous page, ordered top to bottom.
        layers_curr (list[Layer]): Layers from the current page, ordered top to bottom.
        matching_params (dict): Configuration dict with keys.

    Returns:
        OverlapResult | None: indices that define the overlapping layers, or None if no overlap.
    """
    material_threshold = matching_params["duplicate_layer_threshold"]

    # check the longest possible overlap first
    for i in range(min(len(layers_prev), len(layers_curr)), 0, -1):
        match_with_depths_count = 0
        match_ok = True

        for j, (layer_prev, layer_curr) in enumerate(zip(layers_prev[-i:], layers_curr[:i], strict=True)):
            if are_layers_similar(
                layer_prev=layer_prev,
                layer_curr=layer_curr,
                material_threshold=material_threshold,
                is_extremity=(j == 0 or j == i - 1),  # Indicate to function that one layer might be cut (extremities)
            ):
                if layer_prev.depths and layer_prev.depths.start and layer_prev.depths.end:
                    match_with_depths_count += 1
                else:
                    match_with_depths_count = 0
            else:
                if match_with_depths_count >= 2:
                    # allow the overlap, even though not all layers match
                    return OverlapResult(upper_id=len(layers_prev) - i + j, lower_id=j)
                # no valid overlap; break inner loop and go to next value for i
                match_ok = False
                break

        # all layers matched
        if match_ok:
            return OverlapResult(upper_id=len(layers_prev), lower_id=i)


def _is_duplicate(cur_text: str, prev_text: str, threshold: float, is_extremity: bool) -> bool:
    """Detect if layer and prev_layer are duplicates across a page break.

    Strategy:
        - If `is_extremity` is True: uses partial matching to handle truncated boundary layers. Compares
        any suffix of `prev_text` against full `cur_text`, and any prefix of `cur_text` against
        full `prev_text`.
        - If `is_extremity` is False: compares the full text of both layers directly.

    Args:
        cur_text (str): The text of the current layer to compare.
        prev_text (str): The text of the previous layer to compare against.
        threshold (float): The similarity threshold.
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

    return score > threshold
