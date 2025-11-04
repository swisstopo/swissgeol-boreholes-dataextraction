"""This module contains classes for predictions."""

import dataclasses
import logging
from collections import defaultdict
from copy import deepcopy
from typing import TypeVar

from extraction.evaluation.benchmark.metrics import OverallMetricsCatalog
from extraction.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterEvaluator
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.evaluation.metadata_evaluator import MetadataEvaluator
from extraction.features.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwatersInBorehole,
)
from extraction.features.metadata.borehole_name_extraction import NameInDocument
from extraction.features.metadata.coordinate_extraction import Coordinate
from extraction.features.metadata.elevation_extraction import Elevation
from extraction.features.metadata.metadata import BoreholeMetadata
from extraction.features.predictions.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    BoreholeLayersWithGroundTruth,
    BoreholeMetadataWithGroundTruth,
    BoreholePredictions,
    FileGroundwaterWithGroundTruth,
    FileLayersWithGroundTruth,
    FileMetadataWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from extraction.features.stratigraphy.layer.layer import LayersInBorehole, LayersInDocument
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BoreholeListBuilder:
    """Class responsible for constructing a list of BoreholePredictions.

    This is done by aggregating and aligning various extracted document elements.

    This class ensures that all borehole-related lists are extended to match the number of detected boreholes
    and then builds BoreholePredictions objects accordingly.
    """

    def __init__(
        self,
        layers_with_bb_in_document: LayersInDocument,
        file_name: str,
        groundwater_in_doc: GroundwaterInDocument,
        names_in_doc: NameInDocument,
        elevations_list: list[FeatureOnPage[Elevation] | None],
        coordinates_list: list[FeatureOnPage[Coordinate] | None],
    ):
        """Initializes the BoreholeListBuilder with extracted borehole-related data.

        Args:
            layers_with_bb_in_document (LayersInDocument): Object containing a list of extracted boreholes, each
            holding a list of layers and a list of bounding boxes.
            file_name (str): The name of the processed document.
            groundwater_in_doc (GroundwaterInDocument): Contains detected groundwater entries for boreholes.
            names_in_doc (NameInDocument): Contains detected names entries for boreholes.
            elevations_list (list[FeatureOnPage[Elevation] | None]): List of terrain elevation values for detected
                boreholes.
            coordinates_list (list[FeatureOnPage[Coordinate] | None]): List of borehole coordinates.
        """
        self._layers_with_bb_in_document = layers_with_bb_in_document
        self._file_name = file_name
        self._groundwater_in_doc = groundwater_in_doc
        self._names_in_doc = names_in_doc
        self._elevations_list = elevations_list
        self._coordinates_list = coordinates_list

    def build(self) -> list[BoreholePredictions]:
        """Creates a list of BoreholePredictions after, mapping all elements to eachother."""
        if not self._layers_with_bb_in_document.boreholes_layers_with_bb:
            return []  # no borehole identified

        # Removes elevation entries that are also groundwater entries.
        self.remove_elevations_overlaping_groundwater()

        # only criterion for the number of boreholes in the document is the number of borehole layers identified.
        self._num_boreholes = len(self._layers_with_bb_in_document.boreholes_layers_with_bb)

        # for each element, perform the perfect matching with the borehole layers, and retrieve the matching dict
        borehole_idx_to_elevation = self._one_to_one_match_element_to_borehole(self._elevations_list)
        borehole_idx_to_coordinate = self._one_to_one_match_element_to_borehole(self._coordinates_list)
        borehole_idx_to_name = self._one_to_one_match_element_to_borehole(self._names_in_doc.name_feature_list)

        # for groundwater entries, assign each of them to the closest borehole
        borehole_idx_to_list_groundwater = self._many_to_one_match_element_to_borehole(
            self._groundwater_in_doc.groundwater_feature_list
        )

        return [
            BoreholePredictions(
                borehole_index,
                LayersInBorehole(layers_in_borehole_with_bb.predictions),
                self._file_name,
                BoreholeMetadata(
                    borehole_idx_to_elevation[borehole_index],
                    borehole_idx_to_coordinate[borehole_index],
                    borehole_idx_to_name[borehole_index],
                ),
                GroundwatersInBorehole(borehole_idx_to_list_groundwater[borehole_index]),
                layers_in_borehole_with_bb.bounding_boxes,
            )
            for borehole_index, layers_in_borehole_with_bb in enumerate(
                self._layers_with_bb_in_document.boreholes_layers_with_bb
            )
        ]

    @staticmethod
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

    def remove_elevations_overlaping_groundwater(self) -> None:
        """Removes elevation entries that are also groundwater entries.

        This is done by checking if the elevation is present in the groundwater list and removing it if so.
        Some false positives in the groundwater detection causes correct elevations to be deleted (A11462 and A11370).
        To avoid that, we make sure not to delete an elevation entry if it is the only one in the list.
        """
        if len(self._elevations_list) <= 1:
            return
        real_elevations = []
        groundwater_elevations = [gw.feature.elevation for gw in self._groundwater_in_doc.groundwater_feature_list]
        for elevation in self._elevations_list:
            if elevation and elevation.feature.elevation in groundwater_elevations:
                continue
            real_elevations.append(elevation)
        self._elevations_list = real_elevations

    def _many_to_one_match_element_to_borehole(
        self, element_list: list[FeatureOnPage], taken_boreholes: set[int] | None = None
    ) -> dict[int, list[FeatureOnPage]]:
        """Matches extracted elements to boreholes.

        This is done by assigning the clossest borehole to each element.

        Args:
            element_list (list[FeatureOnPage]): list of element to match
            taken_boreholes (set[int]): the set of borehole index that needs to be ignored for the mapping. In this
                context, it is the boreholes that have already been matched (defaults to None).

        Returns:
            dict[int, list[FeatureOnPage]]: the dictionary containing the best mapping borehole_index -> all element
        """
        # solve trivial case
        if self._num_boreholes == 1:
            return {0: element_list}
        # solve case where the list is empty
        if not element_list:
            return {idx: [] for idx in range(self._num_boreholes)}

        # Get a list of references of all borehole bounding boxes
        borehole_bounding_boxes = [
            bh.bounding_boxes for bh in self._layers_with_bb_in_document.boreholes_layers_with_bb
        ]

        if taken_boreholes is None:
            taken_boreholes = set()
        available_boreholes = [idx for idx in range(len(borehole_bounding_boxes)) if idx not in taken_boreholes]

        borehole_index_to_matched_elem = defaultdict(list)
        for feat in element_list:
            best_bbox_idx = min(
                available_boreholes,
                key=lambda j: self._compute_distance(feat, borehole_bounding_boxes[j]),
            )
            borehole_index_to_matched_elem[best_bbox_idx].append(feat)

        return borehole_index_to_matched_elem

    def _one_to_one_match_element_to_borehole(
        self, element_list: list[FeatureOnPage]
    ) -> dict[int, FeatureOnPage | None]:
        """Matches elements (e.g. elevation, coordonates) one-to-one to boreholes based on spatial position.

        The algorithm ensures that each borehole is assigned exactly one element, resolving cases where
        multiple elements might be close to a single borehole. It works iteratively by:

        1. Using a many-to-one matching heuristic to suggest possible element candidates per borehole.
        2. Filtering out elements that have already been assigned.
        3. Selecting the best candidate — defined as the topmost one on the page — if multiple are available.
        4. Repeating until each borehole has a unique match.

        Args:
            element_list (list[FeatureOnPage]): List of extracted elements to match.

        Returns:
            dict[int, FeatureOnPage | None]: Mapping from borehole index to matched element.
        """
        # Ensure there is at least one element for each borehole. This is done by duplicating elements if fewer
        # values were extracted than the number of boreholes, or by filling the list with None values.
        element_list = self._extend_list(element_list, None, self._num_boreholes)

        # solve trivial case and case where the elements are None
        if len(element_list) == 1 or not element_list[0]:
            return {idx: elem for idx, elem in enumerate(element_list)}

        borehole_index_to_matched_elem_index = {}

        # continue until all boreholes are matched
        while len(borehole_index_to_matched_elem_index) != self._num_boreholes:
            # map all elements to their closest borehole.
            borehole_idx_to_many_element_mapping = self._many_to_one_match_element_to_borehole(
                element_list, set(borehole_index_to_matched_elem_index.keys())
            )
            for borehole_index, available_elements in borehole_idx_to_many_element_mapping.items():
                assert borehole_index not in borehole_index_to_matched_elem_index
                assert available_elements
                # if multiple element are bound to the same borehole, always pick the highest on the page
                best_element = min(available_elements, key=lambda elem: (elem.page_number, elem.rect.y0))
                # fill the mapping borehole_index -> element and remove the element from the element list
                borehole_index_to_matched_elem_index[borehole_index] = best_element
                element_list.remove(best_element)

        return borehole_index_to_matched_elem_index

    def _compute_distance(self, feat: FeatureOnPage, bounding_boxes: list[PageBoundingBoxes]) -> float:
        """Computes the distance between a FeatureOnPage objects and the bounding boxes of one borehole."""
        bbox = next((bbox for bbox in bounding_boxes if bbox.page == feat.page_number), None)
        if bbox is None:
            # the current boreholes layers don't appear on the page where the element is
            return float("inf")
        outer_rect = bbox.get_outer_rect()
        element_center = (feat.rect.top_left + feat.rect.bottom_right) / 2
        dist = element_center.distance_to(outer_rect)
        return dist


@dataclasses.dataclass
class AllBoreholePredictionsWithGroundTruth:
    """Class for evaluating all files, after individual boreholes have been match with their ground truth data."""

    predictions_list: list[FilePredictionsWithGroundTruth]

    def evaluate_metadata_extraction(self) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Returns:
            OverallBoreholeMetadataMetrics: the computed metrics for the metadata.
        """
        metadata_list = [
            FileMetadataWithGroundTruth(
                file.filename,
                [
                    BoreholeMetadataWithGroundTruth(
                        predictions.predictions.metadata if predictions.predictions else None,
                        predictions.ground_truth.get("metadata", {}),
                    )
                    for predictions in file.boreholes
                ],
            )
            for file in self.predictions_list
        ]

        return MetadataEvaluator(metadata_list).evaluate()

    def evaluate_geology(self) -> OverallMetricsCatalog:
        """Evaluate the borehole extraction predictions.

        Returns:
            OverallMetricsCatalog: A OverallMetricsCatalog that maps a metrics name to the corresponding
            OverallMetrics object. If no ground truth is available, None is returned.
        """
        languages = set(fp.language for fp in self.predictions_list)
        all_metrics = OverallMetricsCatalog(languages=languages)

        layers_list = [
            FileLayersWithGroundTruth(
                file.filename,
                file.language,
                [
                    BoreholeLayersWithGroundTruth(
                        predictions.predictions.layers_in_borehole if predictions.predictions else None,
                        predictions.ground_truth.get("layers", []),
                    )
                    for predictions in file.boreholes
                ],
            )
            for file in self.predictions_list
        ]
        evaluator = LayerEvaluator(layers_list)
        all_metrics.material_description_metrics = evaluator.get_material_description_metrics()
        all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()
        all_metrics.layer_metrics = evaluator.get_layer_metrics()

        predictions_by_language = {language: [] for language in languages}
        for borehole_data in layers_list:
            # even if metadata can be different for boreholes in the same document, langage is the same (take index 0)
            predictions_by_language[borehole_data.language].append(borehole_data)

        for language, language_predictions_list in predictions_by_language.items():
            evaluator = LayerEvaluator(language_predictions_list)
            setattr(all_metrics, f"{language}_layer_metrics", evaluator.get_layer_metrics())
            setattr(all_metrics, f"{language}_depth_interval_metrics", evaluator.get_depth_interval_metrics())
            setattr(
                all_metrics, f"{language}_material_description_metrics", evaluator.get_material_description_metrics()
            )

        logger.info("Macro avg:")
        logger.info(
            "layer f1: %.1f%%, depth interval f1: %.1f%%, material description f1: %.1f%%",
            all_metrics.layer_metrics.macro_f1() * 100,
            all_metrics.depth_interval_metrics.macro_f1() * 100,
            all_metrics.material_description_metrics.macro_f1() * 100,
        )

        # TODO groundwater should not be in evaluate_geology(), it should be handle by a higher-level function call
        groundwater_list = [
            FileGroundwaterWithGroundTruth(
                file.filename,
                [
                    BoreholeGroundwaterWithGroundTruth(
                        predictions.predictions.groundwater_in_borehole if predictions.predictions else None,
                        predictions.ground_truth.get("groundwater", []) or [],  # value can be `None`
                    )
                    for predictions in file.boreholes
                ],
            )
            for file in self.predictions_list
        ]
        overall_groundwater_metrics = GroundwaterEvaluator(groundwater_list).evaluate()
        all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        all_metrics.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )
        return all_metrics
