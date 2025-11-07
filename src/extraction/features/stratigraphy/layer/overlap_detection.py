"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging

import Levenshtein

from swissgeol_doc_processing.utils.file_utils import read_params

from .layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


config_path = "config"
matching_params = read_params("matching_params.yml", config_path)


def select_boreholes_with_overlap(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
) -> tuple[ExtractedBorehole | None, ExtractedBorehole | None, tuple[int, int] | None]:
    """Remove duplicate layers caused by overlapping scanned pages.

    Compare layers from current page with those from previous page to identify and remove
    duplicates. Layers are compared from bottom to top based on their material descriptions.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): Layers from previous page
        current_page_boreholes (list[ExtractedBorehole]): Layers from current page

    Returns:
        (ExtractedBorehole | None, ExtractedBorehole | None, tuple[int, int] | None):
                The boreholes to be extended, the continuing borehole, and the indexes of the first and last
                duplicated layer in the previous and continuing borehole, if any.
    """
    for current_borehole in current_page_boreholes:
        for previous_page_borehole in previous_page_boreholes:
            bottom_duplicate_idx = find_last_duplicate_layer_index(
                previous_page_borehole.predictions, current_borehole.predictions
            )

            # Potential overlap detected
            if bottom_duplicate_idx is not None:
                # Compute indices to cut duplicate layers at the overlap
                upper_id, lower_id = find_optimal_split(previous_page_borehole, current_borehole, bottom_duplicate_idx)

                return previous_page_borehole, current_borehole, (upper_id, lower_id)

    return None, None, None


def find_optimal_split(
    borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole, last_duplicate_layer_index: int
) -> tuple[int, int]:
    """Find cut indices to remove duplicated layers when merging two boreholes.

    When a borehole continues on the next page, some top layers of the new page
    may duplicate the bottom layers of the previous page. This function walks the
    overlap (from the bottom up) and shifts the split upward if the upper
    layer lacks depth information while the corresponding lower layer has it.
    This preserves the most complete layer data across the boundary.

    Args:
        borehole_to_extend (ExtractedBorehole): The borehole from the previous page.
        borehole_continuation (ExtractedBorehole): The borehole from the current page.
        last_duplicate_layer_index (int): The index of the last duplicated layer in the continuing borehole.

    Returns:
        tuple[int, int]: `(id_upper_end, id_lower_start)` such that concatenating these two slices removes
        duplicates while keeping the most informative depth entries.
    """
    # Check the number of layer that overlaps
    n_overlap = min(len(borehole_to_extend.predictions), last_duplicate_layer_index + 1)
    offset = 0

    # Check if threshold should be moved
    for i in range(n_overlap):
        upper_layer = borehole_to_extend.predictions[-(i + 1)]
        lower_layer = borehole_continuation.predictions[last_duplicate_layer_index - i]
        # It is possible that the information of the upper layers is cut (), check if we can retrieve information from
        # the lower layers
        if not upper_layer.depth_nonempty() and lower_layer.depth_nonempty():
            offset += 1
        else:
            # As soon as information is missing in the lower layer, stop iteration
            break
    # Define cut ids
    id_upper_end = len(borehole_to_extend.predictions) - offset
    id_lower_start = 1 + last_duplicate_layer_index - offset
    return id_upper_end, id_lower_start


def find_last_duplicate_layer_index(previous_page_layers: list[Layer], sorted_layers: list[Layer]) -> int | None:
    """Find the index of the last duplicate layer in the current page compared to the previous page.

    The last duplicated layer is the deepest one that is duplicated, starting from the top.

    Args:
        previous_page_layers (list[Layer]): Layers from a borehole on the previous page sorted from top to bottom
        sorted_layers (list[Layer]): Layers from a borehole on the current page sorted from top to bottom

    Returns:
        int | None: Index of the last duplicate layer or None if not found
    """
    compare_against_idx = len(previous_page_layers) - 1  # begin comparison against the last of previous page
    layer_idx = len(sorted_layers) - 1  # begin from the bottom of the current page, then validate upwards

    bottom_duplicate_idx = None
    duplicate_layer_threshold = matching_params["duplicate_layer_threshold"]

    while layer_idx >= 0:
        current_layer = sorted_layers[layer_idx]
        if not current_layer.material_description:
            layer_idx -= 1
            continue
        if compare_against_idx < 0:
            break  # we've run out of previous layers to compare against

        previous_page_layer = previous_page_layers[compare_against_idx]

        # 1. check of the current layer with the compare_against_idx'th layer of previous page.
        if _is_duplicate(
            current_layer.material_description.text,
            previous_page_layer.material_description.text,
            duplicate_layer_threshold,
        ):
            # 2. check in case of wrong lines split of the merged layer with the compare_against_idx'th layer of
            # previous page.
            layer_idx_delta = 1
            if layer_idx >= 1 and sorted_layers[layer_idx - 1].material_description:
                text_merged = " ".join(
                    [sorted_layers[layer_idx - 1].material_description.text, current_layer.material_description.text]
                )
                if _is_duplicate(
                    text_merged, previous_page_layer.material_description.text, duplicate_layer_threshold
                ):
                    # If the merged is also a duplicate, we should skip both layers.
                    layer_idx_delta = 2

            if bottom_duplicate_idx is None:
                # record the lowest layer that is duplicated (i.e. the first encountered, as we iterate backwards).
                bottom_duplicate_idx = layer_idx
            layer_idx -= layer_idx_delta  # skip the wrongly split layer(s)
            compare_against_idx -= 1
            continue

        # 3. No duplicate was found, we just continue with the above layer
        if bottom_duplicate_idx is None:
            layer_idx -= 1
            continue
        # 4. No duplicate was found, but we previously found a duplicate (bottom_duplicate_idx was set), it likely
        # was a false positive and we reset the search
        layer_idx = bottom_duplicate_idx - 1  # just above the false positive
        bottom_duplicate_idx = None
        compare_against_idx = len(previous_page_layers) - 1  # reset with the last layer of the previous page

    return bottom_duplicate_idx


def _is_duplicate(cur_text: str, prev_text: str, t: float):
    """Detect if layer and prev_layer are duplicates across a page break.

    Strategy:
      - Check any suffix of prev_text words vs the full current text.
      - Check any prefix of current words vs the full prev_text text.
    This covers both split cases (the prev_text was cut off / the cur_text only repeats the tail).

    Args:
        cur_text (str): The text of the current layer to compare.
        prev_text (str): The text of the previous layer to compare against.
        t (float): The similarity threshold.

    Returns:
        bool: True if the layers are considered duplicates, False otherwise.
    """
    cur_text = cur_text.lower()
    prev_text = prev_text.lower()
    min_length = min(len(cur_text), len(prev_text))
    return (
        Levenshtein.ratio(cur_text, prev_text[-min_length:]) > t
        or Levenshtein.ratio(cur_text[:min_length], prev_text) > t
    )
