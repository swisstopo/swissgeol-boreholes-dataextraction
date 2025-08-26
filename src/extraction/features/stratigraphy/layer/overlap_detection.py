"""This module contains functionality for detecting duplicate layers across pdf pages."""

import logging

import Levenshtein

from .layer import ExtractedBorehole, Layer

logger = logging.getLogger(__name__)


def remove_scan_overlap_layers(
    current_page_index: int,
    previous_layers_with_bb: list[ExtractedBorehole],
    current_layers_with_bb: list[ExtractedBorehole],
) -> list[ExtractedBorehole]:
    """Remove duplicate layers caused by overlapping scanned pages.

    Compare layers from current page with those from previous page to identify and remove
    duplicates. Layers are compared from bottom to top based on their material descriptions.

    Args:
        current_page_index (int): Current page index
        previous_layers_with_bb (list[ExtractedBorehole]): Layers from previous page
        current_layers_with_bb (list[ExtractedBorehole]): Layers from current page

    Returns:
        list[ExtractedBorehole]: Current page layers with duplicates removed
    """
    if current_page_index == 0:
        return current_layers_with_bb

    previous_page_layers = [lay for boreholes in previous_layers_with_bb for lay in boreholes.predictions]
    if not previous_page_layers:
        return current_layers_with_bb

    non_duplicated_extracted_boreholes: list[ExtractedBorehole] = []
    for current_borehole in current_layers_with_bb:
        assert (
            len({page for layer in current_borehole.predictions for page in layer.material_description.pages}) == 1
        ), "At this point, all layers should be on the same page."
        sorted_layers = sorted(
            current_borehole.predictions, key=lambda x: x.material_description.rects_with_pages[0].rect.y0
        )

        bottom_duplicate_idx = find_last_duplicate_layer_index(previous_page_layers, sorted_layers)

        non_duplicated_extracted_boreholes.append(
            ExtractedBorehole(
                predictions=sorted_layers[bottom_duplicate_idx + 1 :]
                if bottom_duplicate_idx is not None
                else sorted_layers,
                bounding_boxes=current_borehole.bounding_boxes,
            )
        )
    return non_duplicated_extracted_boreholes


def find_last_duplicate_layer_index(previous_page_layers: list[Layer], sorted_layers: list[Layer]) -> int | None:
    """Find the index of the last duplicate layer in the current page compared to the previous page.

    The last duplicated layer is the deepest one that is duplicated, starting from the top.

    Args:
        previous_page_layers (list[Layer]): Layers from previous page sorted from top to bottom
        sorted_layers (list[Layer]): Layers from current page sorted from top to bottom

    Returns:
        int | None: Index of the last duplicate layer or None if not found
    """
    compare_against_idx = -1  # begin comparison against the last of previous page
    layer_idx = len(sorted_layers) - 1  # begin from the bottom of the current page, then validate upwards

    bottom_duplicate_idx = None

    while layer_idx >= 0:
        current_layer = sorted_layers[layer_idx]
        if not current_layer.material_description:
            layer_idx -= 1
            continue
        if abs(compare_against_idx) > len(previous_page_layers):
            break  # we've run out of previous layers to compare against

        previous_page_layer = previous_page_layers[compare_against_idx]

        # 1. check of the current layer with the compare_against_idx'th layer of previous page.
        if _is_duplicate(current_layer.material_description.text, previous_page_layer.material_description.text):
            # 2. check in case of wrong lines split of the merged layer with the compare_against_idx'th layer of
            # previous page.
            text_merged = ""
            if layer_idx >= 1 and sorted_layers[layer_idx - 1].material_description:
                text_merged = (
                    sorted_layers[layer_idx - 1].material_description.text
                    + " "
                    + current_layer.material_description.text
                )
            if _is_duplicate(text_merged, previous_page_layer.material_description.text):
                # If the merged is also a duplicate, we should skip both layers.
                if bottom_duplicate_idx is None:
                    # record the lowest layer that is duplicated (i.e. the first encountered, as we iterate backwards).
                    bottom_duplicate_idx = layer_idx
                layer_idx -= 2  # skip the wrongly split layer
                compare_against_idx -= 1
                continue

            # If the merge is not a duplicated, this is the standart case, and we continue verifying the layers above.
            if bottom_duplicate_idx is None:
                bottom_duplicate_idx = layer_idx
            layer_idx -= 1
            compare_against_idx -= 1
            continue

        # 3. No duplicate was found, we just continue with the above layer
        if bottom_duplicate_idx is None:
            layer_idx -= 1
            continue
        # 4. No duplicate was found, but we previously found a duplicate (bottom_duplicate_idx was set), it likelly
        # was a false positive and we reset the search
        layer_idx = bottom_duplicate_idx - 1  # just above the false positive
        bottom_duplicate_idx = None
        compare_against_idx = -1  # reset with the last layer of the previous page

    return bottom_duplicate_idx


def _is_duplicate(cur_text: str, prev_text: str, t=0.95):
    """Detect if layer and prev_layer are duplicates across a page break.

    Strategy:
      - Check any suffix of prev_text words vs the full current text.
      - Check any prefix of current words vs the full prev_text text.
    This covers both split cases (the prev_text was cut off / the cur_text only repeats the tail).

    Args:
        cur_text (str): The text of the current layer to compare.
        prev_text (str): The text of the previous layer to compare against.
        t (float): The similarity threshold (default is 0.95).

    Returns:
        bool: True if the layers are considered duplicates, False otherwise.
    """
    cur_text = cur_text.lower()
    cur_words = cur_text.split()
    prev_text = prev_text.lower()
    prev_words = prev_text.split()
    return any(Levenshtein.ratio(cur_text, " ".join(prev_words[i:])) > t for i in range(len(prev_words))) or any(
        Levenshtein.ratio(" ".join(cur_words[: i + 1]), prev_text) > t for i in range(len(cur_words))
    )
