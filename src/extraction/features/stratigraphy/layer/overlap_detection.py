"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging
import math
import re
import unicodedata
from dataclasses import dataclass

import Levenshtein

from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)

MAX_BOUNDARY_LAYERS_TO_DROP = 2
DEPTH_VALUE_TOLERANCE = 0.05
MIN_DEPTH_OVERLAP_MATCHES = 2


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
    material_threshold: float = 0.90,
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
        material_threshold (float): Minimum similarity threshold for material descriptions. Defaults to 0.90.
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

    Tries a plain match first. `_find_longest_overlap`'s window always starts its comparison at
    `layers_curr[0]` (and ends at `layers_prev[-1]`), so a boundary layer that doesn't compare well to
    its counterpart - e.g. because table/OCR parsing collapsed several real rows into it, or
    truncated/paraphrased its text - blocks every window from aligning, hiding a genuine overlap in
    the rest of the page. If the plain match fails, retry with up to `MAX_BOUNDARY_LAYERS_TO_DROP`
    layers dropped from either end.

    Args:
        layers_prev (list[Layer]): Layers from the previous page, ordered top to bottom.
        layers_curr (list[Layer]): Layers from the current page, ordered top to bottom.
        matching_params (dict): Configuration dict with keys.

    Returns:
        OverlapResult | None: indices that define the overlapping layers, or None if no overlap.
    """
    result = _find_longest_overlap(layers_prev, layers_curr, matching_params)
    if result is not None:
        return result

    for drop in range(1, MAX_BOUNDARY_LAYERS_TO_DROP + 1):
        if len(layers_curr) <= drop:
            break
        trimmed_curr = layers_curr[drop:]
        result = _find_longest_overlap(layers_prev, trimmed_curr, matching_params)
        if result is not None and _is_trustworthy_retry_match(result, trimmed_curr):
            return OverlapResult(upper_id=result.upper_id, lower_id=result.lower_id + drop)

    for drop in range(1, MAX_BOUNDARY_LAYERS_TO_DROP + 1):
        if len(layers_prev) <= drop:
            break
        trimmed_prev = layers_prev[:-drop]
        result = _find_longest_overlap(trimmed_prev, layers_curr, matching_params)
        if result is not None and _is_trustworthy_retry_match(result, layers_curr):
            return result

    return _find_depth_reset_overlap(layers_prev, layers_curr)


def _is_trustworthy_retry_match(result: OverlapResult, trimmed_curr: list[Layer]) -> bool:
    """Guard against accepting a boundary-drop retry on a single coincidental text match.

    Dropping boundary layers to force an alignment is inherently speculative - unlike a match found
    without dropping anything, it discards content to make things fit. A match spanning several
    layers is trustworthy regardless of depth info (consistent with the plain, undropped search). A
    match of just one layer needs the extra corroboration of a real, matching depth interval - text
    similarity alone on a single row is too easy to hit by coincidence (e.g. a short, generic
    description repeated elsewhere in the document).

    Args:
        result (OverlapResult): The result from `_find_longest_overlap` on the trimmed layers.
        trimmed_curr (list[Layer]): The (already boundary-trimmed) current-page layers that were matched.

    Returns:
        bool: True if the match is trustworthy enough to accept.
    """
    if result.lower_id > 1:
        return True
    matched_layer = trimmed_curr[0]
    return bool(matched_layer.depths and matched_layer.depths.start and matched_layer.depths.end)


def _collect_depth_values(layers: list[Layer]) -> list[float]:
    """Collect all known depth boundary values (layer starts and ends), in top-to-bottom order.

    Args:
        layers (list[Layer]): Layers to collect boundary values from.

    Returns:
        list[float]: The depth values, in the order they appear.
    """
    values = []
    for layer in layers:
        if not layer.depths:
            continue
        if layer.depths.start and layer.depths.start.value is not None:
            values.append(layer.depths.start.value)
        if layer.depths.end and layer.depths.end.value is not None:
            values.append(layer.depths.end.value)
    return values


def _find_depth_reset_overlap(layers_prev: list[Layer], layers_curr: list[Layer]) -> OverlapResult | None:
    """Find an overlap by depth values alone, for boundaries where the OCR'd text can't be trusted.

    Some scans have a physical ruler or fold obscuring the material description text right where a
    borehole continues onto the next page, so the text-based matches above find nothing there. When
    that happens, the current page's depths restart lower than the previous page's last depth -
    looking like a new, shallower borehole - but several of those depth values are the exact same
    ones already seen on the previous page. That repetition is strong evidence that this is a
    re-scanned duplicate of already-seen content, not a new borehole.

    Args:
        layers_prev (list[Layer]): Layers from the previous page, ordered top to bottom.
        layers_curr (list[Layer]): Layers from the current page, ordered top to bottom.

    Returns:
        OverlapResult | None: indices that define the overlapping layers, or None if no overlap.
    """
    prev_values = _collect_depth_values(layers_prev)
    if not prev_values:
        return None
    prev_max = max(prev_values)

    lower_id = 0
    matches = 0
    for layer in layers_curr:
        end = layer.depths.end.value if layer.depths and layer.depths.end else None
        if end is None or end > prev_max + DEPTH_VALUE_TOLERANCE:
            break
        if any(math.isclose(end, value, abs_tol=DEPTH_VALUE_TOLERANCE) for value in prev_values):
            matches += 1
        lower_id += 1

    if matches < MIN_DEPTH_OVERLAP_MATCHES:
        return None
    return OverlapResult(upper_id=len(layers_prev), lower_id=lower_id)


def _find_longest_overlap(
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
                if (
                    layer_prev.depths
                    and layer_prev.depths.start
                    and layer_prev.depths.end
                    and layer_curr.depths
                    and layer_curr.depths.start
                    and layer_curr.depths.end
                ):
                    match_with_depths_count += 1
                else:
                    match_with_depths_count = 0
            else:
                if match_with_depths_count >= 1:
                    # allow the overlap, even though not all layers match
                    return OverlapResult(upper_id=len(layers_prev) - i + j, lower_id=j)
                # no valid overlap; break inner loop and go to next value for i
                match_ok = False
                break

        # all layers matched
        if match_ok:
            return OverlapResult(upper_id=len(layers_prev), lower_id=i)


def _normalize_for_comparison(text: str) -> str:
    """Fold accents and collapse punctuation/whitespace noise so independent OCR passes compare as equal.

    Two scans of the same physical row commonly disagree only on comma placement, spacing, or an
    accented character (e.g. "MERGEL, DUNKELGRÜN" vs "MERGEL DUNKELGRUN") - noise that otherwise drops
    the Levenshtein ratio just below the similarity threshold.

    Args:
        text (str): The raw material description text.

    Returns:
        str: Lowercased text with accents folded to their base letter and punctuation/whitespace collapsed.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


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
    cur_text = _normalize_for_comparison(cur_text)
    prev_text = _normalize_for_comparison(prev_text)

    if is_extremity:
        min_length = min(len(cur_text), len(prev_text))
        score = max(
            Levenshtein.ratio(cur_text, prev_text[-min_length:]),
            Levenshtein.ratio(cur_text[:min_length], prev_text),
        )
    else:
        score = Levenshtein.ratio(cur_text, prev_text)

    return score > threshold
