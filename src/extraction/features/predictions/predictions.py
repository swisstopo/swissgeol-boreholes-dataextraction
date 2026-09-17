"""This module contains classes for predictions."""

import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeVar

from extraction.features.extracted_borehole import ExtractedBorehole
from extraction.features.groundwater.groundwater import Groundwater
from extraction.features.metadata.borehole_name_extraction import BoreholeName
from extraction.features.metadata.coordinate_extraction import Coordinate
from extraction.features.metadata.elevation_extraction import Elevation
from extraction.features.predictions.borehole_predictions import (
    BoreholePredictions,
)
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _extend_list(lst: list[T], default_elem: T, target_length: int) -> list[T]:
    """Extends a list with deep copies of a base element until it reaches the target length.

    deepcopy is necessary, because the is_correct attribute is already stored on this object, but the same
    extracted value might be correct on one borehole and incorrect on another one.
    """

    def create_new_elem():
        return deepcopy(lst[0]) if lst else default_elem

    while len(lst) < target_length:
        lst.append(create_new_elem())  # Append copies to match the required length

    return lst


def _compute_distance(feat: FeatureOnPage, borehole: ExtractedBorehole) -> float | None:
    """Computes the distance between a FeatureOnPage object and the bounding boxes of one borehole."""
    bbox = next((bbox for bbox in borehole.bounding_boxes if bbox.page == feat.page_number), None)
    if bbox is None:
        # the current borehole's layers don't appear on the page where the element is
        return None
    outer_rect = bbox.get_outer_rect()
    element_center = (feat.rect.top_left + feat.rect.bottom_right) / 2
    return element_center.distance_to(outer_rect)


def _many_to_one_match_element_to_borehole(
    element_list: list[FeatureOnPage],
    boreholes: list[ExtractedBorehole],
    taken_boreholes: set[int] | None = None,
) -> dict[int, list[FeatureOnPage]]:
    """Matches extracted elements to boreholes.

    This is done by assigning the closest borehole to each element.

    Args:
        element_list (list[FeatureOnPage]): list of element to match
        boreholes (list[ExtractedBorehole]): the boreholes to match the elements against.
        taken_boreholes (set[int]): the set of borehole index that needs to be ignored for the mapping. In this
            context, it is the boreholes that have already been matched (defaults to None).

    Returns:
        dict[int, list[FeatureOnPage]]: the dictionary containing the best mapping borehole_index -> all element
    """
    # solve trivial case
    if len(boreholes) == 1:
        return {0: element_list}
    # solve case where the list is empty
    if not element_list:
        return {idx: [] for idx in range(len(boreholes))}

    if taken_boreholes is None:
        taken_boreholes = set()
    available_boreholes = [idx for idx in range(len(boreholes)) if idx not in taken_boreholes]

    borehole_index_to_matched_elem = defaultdict(list)
    for feat in element_list:
        # Compute distance between feature and borehole
        distances = {j: _compute_distance(feat, boreholes[j]) for j in available_boreholes}
        # Filter candidates based on valid distance
        candidates_idx = [j for j, d in distances.items() if d is not None]
        # Check if at least one valid candidate
        if not candidates_idx:
            continue
        # Find best candidate
        best_bbox_idx = min(candidates_idx, key=lambda j: distances[j])

        # Distance should not be infinite
        borehole_index_to_matched_elem[best_bbox_idx].append(feat)

    return borehole_index_to_matched_elem


def _one_to_one_match_element_to_borehole(
    element_list: list[FeatureOnPage], boreholes: list[ExtractedBorehole]
) -> dict[int, FeatureOnPage | None]:
    """Matches elements (e.g. elevation, coordinates) one-to-one to boreholes based on spatial position.

    The algorithm ensures that each borehole is assigned exactly one element, resolving cases where
    multiple elements might be close to a single borehole. It works iteratively by:

    1. Using a many-to-one matching heuristic to suggest possible element candidates per borehole.
    2. Filtering out elements that have already been assigned.
    3. Selecting the best candidate — defined as the topmost one on the page — if multiple are available.
    4. Repeating until each borehole has a unique match.

    Args:
        element_list (list[FeatureOnPage]): List of extracted elements to match.
        boreholes (list[ExtractedBorehole]): the boreholes to match the elements against.

    Returns:
        dict[int, FeatureOnPage | None]: Mapping from borehole index to matched element.
    """
    num_boreholes = len(boreholes)

    # Ensure there is at least one element for each borehole. This is done by duplicating elements if fewer
    # values were extracted than the number of boreholes, or by filling the list with None values.
    element_list = _extend_list(element_list, None, num_boreholes)

    # solve trivial case and case where the elements are None
    if len(element_list) == 1 or not element_list[0]:
        return {idx: elem for idx, elem in enumerate(element_list)}

    borehole_index_to_matched_elem_index = {}
    # continue until all boreholes are matched

    while len(borehole_index_to_matched_elem_index) != num_boreholes:
        # map all elements to their closest borehole.
        borehole_idx_to_many_element_mapping = _many_to_one_match_element_to_borehole(
            element_list, boreholes, set(borehole_index_to_matched_elem_index.keys())
        )

        # No more potential matching found, break rule
        if not borehole_idx_to_many_element_mapping:
            break

        # Iterate over mapping to find best candidates in list
        for borehole_index, available_elements in borehole_idx_to_many_element_mapping.items():
            assert borehole_index not in borehole_index_to_matched_elem_index
            assert available_elements
            # if multiple element are bound to the same borehole, always pick the highest on the page
            best_element = min(available_elements, key=lambda elem: (elem.page_number, elem.rect.y0))
            # fill the mapping borehole_index -> element and remove the element from the element list
            borehole_index_to_matched_elem_index[borehole_index] = best_element
            element_list.remove(best_element)

    return borehole_index_to_matched_elem_index


def _remove_elevations_matching_groundwater(
    elevation_entries: list[FeatureOnPage[Elevation]],
    groundwater_entries: list[FeatureOnPage[Groundwater]],
) -> list[FeatureOnPage[Elevation]]:
    """Removes elevation entries that are also groundwater entries.

    Some false positives in the groundwater detection causes correct elevations to be deleted (A11462 and A11370).
    To avoid that, we make sure not to delete an elevation entry if it is the only one found on this page.
    """
    if len(elevation_entries) <= 1:
        return elevation_entries
    groundwater_elevations = [gw.feature.elevation for gw in groundwater_entries]
    return [elevation for elevation in elevation_entries if elevation.feature.elevation not in groundwater_elevations]


def assign_page_metadata(extracted_boreholes: list[ExtractedBorehole], candidates: "PageMetadataCandidates") -> None:
    """Matches this page's (or these pages') metadata candidates to these boreholes, in place.

    `extracted_boreholes` is normally the list of boreholes found on a single page, but during
    adjacent-page resolution it can also be a single neighboring page's boreholes matched against
    another page's leftover candidates.
    """
    if not extracted_boreholes:
        return
    elevation_entries = _remove_elevations_matching_groundwater(candidates.elevations, candidates.groundwater)

    name_by_borehole = _one_to_one_match_element_to_borehole(candidates.names, extracted_boreholes)
    elevation_by_borehole = _one_to_one_match_element_to_borehole(elevation_entries, extracted_boreholes)
    coordinate_by_borehole = _one_to_one_match_element_to_borehole(candidates.coordinates, extracted_boreholes)
    groundwater_by_borehole = _many_to_one_match_element_to_borehole(candidates.groundwater, extracted_boreholes)

    for index, borehole in enumerate(extracted_boreholes):
        borehole.metadata.name = borehole.metadata.name or name_by_borehole.get(index)
        borehole.metadata.elevation = borehole.metadata.elevation or elevation_by_borehole.get(index)
        borehole.metadata.coordinates = borehole.metadata.coordinates or coordinate_by_borehole.get(index)
        borehole.groundwater.features.extend(groundwater_by_borehole.get(index, []))


@dataclass
class PageMetadataCandidates:
    """Metadata candidates found on one page, deferred because it had no borehole to match against."""

    page_index: int
    names: list[FeatureOnPage[BoreholeName]] = field(default_factory=list)
    elevations: list[FeatureOnPage[Elevation]] = field(default_factory=list)
    coordinates: list[FeatureOnPage[Coordinate]] = field(default_factory=list)
    groundwater: list[FeatureOnPage[Groundwater]] = field(default_factory=list)


def resolve_boreholeless_pages(
    boreholes_per_page: list[list[ExtractedBorehole]],
    boreholeless_pages: list[PageMetadataCandidates],
) -> None:
    """Attaches metadata found on a borehole-less page to an adjacent page's borehole, in place.

    Metadata (name, coordinates, elevation, groundwater) is sometimes printed on its own page, separate
    from the stratigraphy table (e.g. a cover page). Such a page has no borehole of its own to match
    against. If exactly one of its two neighboring pages has exactly one borehole, the metadata is
    attached there. If neither or both do, which borehole it belongs to is ambiguous, so it is left
    unassigned rather than guessed.
    """
    for page in boreholeless_pages:
        neighbor_indices = (page.page_index - 1, page.page_index + 1)
        unambiguous_neighbors = [
            boreholes_per_page[i]
            for i in neighbor_indices
            if 0 <= i < len(boreholes_per_page) and len(boreholes_per_page[i]) == 1
        ]
        if len(unambiguous_neighbors) == 1:
            assign_page_metadata(unambiguous_neighbors[0], page)


def build_borehole_predictions(extracted_boreholes: list[ExtractedBorehole]) -> list[BoreholePredictions]:
    """Builds the final list of BoreholePredictions from already-matched, merged boreholes.

    Metadata (name, elevation, coordinates, groundwater) is matched to boreholes per page, before
    continuation-detection merges boreholes across pages (see `assign_page_metadata` /
    `resolve_boreholeless_pages` in `extract.py`), so by the time this runs each merged borehole already
    carries its own metadata; this only reshapes it into the output type.
    """
    return [
        BoreholePredictions(
            borehole_index,
            borehole.predictions,
            borehole.metadata,
            borehole.groundwater,
            borehole.bounding_boxes,
        )
        for borehole_index, borehole in enumerate(extracted_boreholes)
    ]
