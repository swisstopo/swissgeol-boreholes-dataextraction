"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging

import Levenshtein

from swissgeol_doc_processing.utils.file_utils import read_params

from .layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


matching_params = read_params("matching_params.yml")


def select_boreholes_with_scan_overlap(
    previous_page_boreholes: list[ExtractedBorehole],
    current_page_boreholes: list[ExtractedBorehole],
) -> tuple[ExtractedBorehole | None, ExtractedBorehole | None, int | None]:
    """Remove duplicate layers caused by overlapping scanned pages.

    Compare layers from current page with those from previous page to identify and remove
    duplicates. Layers are compared from bottom to top based on their material descriptions.

    Args:
        previous_page_boreholes (list[ExtractedBorehole]): Layers from previous page
        current_page_boreholes (list[ExtractedBorehole]): Layers from current page

    Returns:
        (ExtractedBorehole | None, ExtractedBorehole | None, int | None):
                The boreholes to be extended, the continuing borehole, and the index of the last duplicated
                layer in the continuing borehole, if any.
    """
    for current_borehole in current_page_boreholes:
        for previous_page_borehole in previous_page_boreholes:
            bottom_duplicate_idx = find_last_duplicate_layer_index(
                previous_page_borehole.predictions, current_borehole.predictions
            )

            if bottom_duplicate_idx is not None:
                return previous_page_borehole, current_borehole, bottom_duplicate_idx
    return None, None, None


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
