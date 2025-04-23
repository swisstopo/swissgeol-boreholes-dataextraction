"""This module contains classes for predictions."""

import dataclasses
import logging
from collections import defaultdict
from copy import deepcopy
from typing import TypeVar

import numpy as np
from extraction.evaluation.benchmark.metrics import OverallMetricsCatalog
from extraction.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from extraction.evaluation.groundwater_evaluator import GroundwaterEvaluator
from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.evaluation.metadata_evaluator import MetadataEvaluator
from extraction.extraction.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwatersInBorehole,
)
from extraction.extraction.metadata.coordinate_extraction import Coordinate
from extraction.extraction.metadata.elevation_extraction import Elevation
from extraction.extraction.metadata.metadata import BoreholeMetadata
from extraction.extraction.predictions.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    BoreholeLayersWithGroundTruth,
    BoreholeMetadataWithGroundTruth,
    BoreholePredictions,
    FileGroundwaterWithGroundTruth,
    FileLayersWithGroundTruth,
    FileMetadataWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from extraction.extraction.stratigraphy.layer.layer import LayersInBorehole, LayersInDocument
from extraction.extraction.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from extraction.extraction.utils.data_extractor import FeatureOnPage
from scipy.optimize import linear_sum_assignment

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
        elevations_list: list[FeatureOnPage[Elevation] | None],
        coordinates_list: list[FeatureOnPage[Coordinate] | None],
    ):
        """Initializes the BoreholeListBuilder with extracted borehole-related data.

        Args:
            layers_with_bb_in_document (LayersInDocument): Object containing a list of extracted boreholes, each
            holding a list of layers and a list of bounding boxes.
            file_name (str): The name of the processed document.
            groundwater_in_doc (GroundwaterInDocument): Contains detected groundwater entries for boreholes.

            elevations_list (list[FeatureOnPage[Elevation] | None]): List of terrain elevation values for detected
                boreholes.
            coordinates_list (list[FeatureOnPage[Coordinate] | None]): List of borehole coordinates.
        """
        self._layers_with_bb_in_document = layers_with_bb_in_document
        self._file_name = file_name
        self._groundwater_in_doc = groundwater_in_doc
        self._elevations_list = elevations_list
        self._coordinates_list = coordinates_list

    def build(self) -> list[BoreholePredictions]:
        """Creates a list of BoreholePredictions after, mapping all elements to eachother."""
        if not self._layers_with_bb_in_document.boreholes_layers_with_bb:
            return []  # no borehole identified
        self._extend_lenght_metadata_elements()

        # for each element, perform the perfect matching with the borehole layers, and retrieve the matching dict
        borehole_to_elevation_map = self._one_to_one_match_element_to_borehole(self._elevations_list)
        borehole_to_coordinate_map = self._one_to_one_match_element_to_borehole(self._coordinates_list)
        borehole_to_list_groundwater_map = self._many_to_one_match_element_to_borehole(
            self._groundwater_in_doc.groundwater_feature_list
        )

        return [
            BoreholePredictions(
                borehole_index,
                LayersInBorehole(layers_in_borehole_with_bb.predictions),
                self._file_name,
                BoreholeMetadata(
                    self._elevations_list[borehole_to_elevation_map[borehole_index]],
                    self._coordinates_list[borehole_to_coordinate_map[borehole_index]],
                ),
                GroundwatersInBorehole(
                    [
                        self._groundwater_in_doc.groundwater_feature_list[idx]
                        for idx in borehole_to_list_groundwater_map[borehole_index]
                    ]
                ),
                layers_in_borehole_with_bb.bounding_boxes,
            )
            for borehole_index, layers_in_borehole_with_bb in enumerate(
                self._layers_with_bb_in_document.boreholes_layers_with_bb
            )
        ]

    def _extend_lenght_metadata_elements(self) -> None:
        """Ensures that all lists have at least the same length by duplicating elements if necessary."""
        num_boreholes = len(self._layers_with_bb_in_document.boreholes_layers_with_bb)

        self._elevations_list = self._extend_list(self._elevations_list, None, num_boreholes)
        self._coordinates_list = self._extend_list(self._coordinates_list, None, num_boreholes)

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

    def _many_to_one_match_element_to_borehole(self, element_list: list[FeatureOnPage]) -> dict[int, list[int]]:
        """Matches extracted elements to boreholes.

        This is done by assigning the clossest borehole to each element.

        Args:
            element_list (list[FeatureOnPage]): list of element to match

        Returns:
            dict[int, list[int]]: the dictionary containing the best mapping borehole_index -> all element_indexes
        """
        num_boreholes = len(self._layers_with_bb_in_document.boreholes_layers_with_bb)
        # solve trivial case
        if num_boreholes == 1:
            return {0: [idx for idx in range(len(element_list))]}
        # solve case where the list is empty
        if not element_list:
            return {idx: [] for idx in range(num_boreholes)}

        # Get a list of references of all borehole bounding boxes
        borehole_bounding_boxes = [
            bh.bounding_boxes for bh in self._layers_with_bb_in_document.boreholes_layers_with_bb
        ]

        borehole_index_to_matched_elem_index = defaultdict(list)
        for i, feat in enumerate(element_list):
            best_bbox_idx = min(
                range(len(borehole_bounding_boxes)),
                key=lambda j: self._compute_distance(feat, borehole_bounding_boxes[j]),
            )
            borehole_index_to_matched_elem_index[best_bbox_idx].append(i)

        return borehole_index_to_matched_elem_index

    def _one_to_one_match_element_to_borehole(
        self,
        element_list: list[FeatureOnPage | None],
    ) -> dict[int, int]:
        """Matches extracted elements to boreholes.

        This is done by minimizing the total sum of the distances from the elements to the corresponding layers.

        Args:
            element_list (list[FeatureOnPage  |  None ]): list of element to match

        Returns:
            dict[int, int]: the dictionary containing the best mapping borehole_index -> element_index
        """
        # solve trivial case and case where the elements are None
        if len(element_list) == 1 or not element_list[0]:
            return {idx: idx for idx in range(len(element_list))}

        # Get a list of references of all borehole bounding boxes
        borehole_bounding_boxes = [
            bh.bounding_boxes for bh in self._layers_with_bb_in_document.boreholes_layers_with_bb
        ]

        cost_matrix = np.zeros((len(element_list), len(borehole_bounding_boxes)))

        # Fill the cost matrix with distances between each element and each borehole
        for i, feat in enumerate(element_list):
            for j, bboxes in enumerate(borehole_bounding_boxes):
                dist = self._compute_distance(feat, bboxes)  # Compute the distance
                cost_matrix[i, j] = dist

        # Use the Hungarian algorithm (Kuhn-Munkres) to solve the assignment problem.
        # It finds the optimal one-to-one matching between elements and borehole layers such that the total
        # distance (cost) between matched pairs is minimized.
        # In simpler terms: out of all possible ways to pair elements with boreholes, this picks the combination that
        # results in the shortest total distance between them.
        elem_indexes, layer_indexes = linear_sum_assignment(cost_matrix)

        # Store the index of the matched elements in a dictionary
        borehole_index_to_matched_elem_index = {}
        for elem_idx, layer_idx in zip(elem_indexes, layer_indexes, strict=True):
            borehole_index_to_matched_elem_index[layer_idx] = elem_idx

        return borehole_index_to_matched_elem_index

    def _compute_distance(self, feat: FeatureOnPage, bounding_boxes: list[PageBoundingBoxes]) -> float:
        """Computes the distance between a FeatureOnPage objects and the bounding boxes of one borehole."""
        bbox = next((bbox for bbox in bounding_boxes if bbox.page == feat.page), None)
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
        all_metrics.layer_metrics = evaluator.get_layer_metrics()
        all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()

        predictions_by_language = {language: [] for language in languages}
        for borehole_data in layers_list:
            # even if metadata can be different for boreholes in the same document, langage is the same (take index 0)
            predictions_by_language[borehole_data.language].append(borehole_data)

        for language, language_predictions_list in predictions_by_language.items():
            evaluator = LayerEvaluator(language_predictions_list)
            setattr(
                all_metrics,
                f"{language}_layer_metrics",
                evaluator.get_layer_metrics(),
            )
            setattr(all_metrics, f"{language}_depth_interval_metrics", evaluator.get_depth_interval_metrics())

        logger.info("Macro avg:")
        logger.info(
            "F1: %.1f%%, precision: %.1f%%, recall: %.1f%%, depth_interval_precision: %.1f%%",
            all_metrics.layer_metrics.macro_f1() * 100,
            all_metrics.layer_metrics.macro_precision() * 100,
            all_metrics.layer_metrics.macro_recall() * 100,
            all_metrics.depth_interval_metrics.macro_precision() * 100,
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
