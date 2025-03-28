"""This module contains classes for predictions."""

import dataclasses
import logging
from copy import deepcopy
from typing import TypeVar

import fitz
import numpy as np
from scipy.optimize import linear_sum_assignment

from stratigraphy.benchmark.metrics import OverallMetricsCatalog
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.bounding_boxes import PageBoundingBoxes
from stratigraphy.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInDocument, GroundwatersInBorehole
from stratigraphy.layer.layer import LayersInBorehole, LayersInDocument
from stratigraphy.metadata.coordinate_extraction import Coordinate
from stratigraphy.metadata.elevation_extraction import Elevation
from stratigraphy.metadata.metadata import BoreholeMetadata
from stratigraphy.util.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    BoreholeLayersWithGroundTruth,
    BoreholeMetadataWithGroundTruth,
    BoreholePredictions,
    FileGroundwaterWithGroundTruth,
    FileLayersWithGroundTruth,
    FileMetadataWithGroundTruth,
    FilePredictionsWithGroundTruth,
)

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
        """Creates a list of BoreholePredictions after ensuring all lists have the same length."""
        if not self._layers_with_bb_in_document.boreholes_layers_with_bb:
            return []  # no borehole identified
        self._extend_length_to_match_boreholes_num_pred()

        # for each element, perform the perfect matching with the borehole layers, and retrieve the matching dict
        borehole_to_elevation_map = self._match_element_to_borehole(self._elevations_list)
        borehole_to_coordinate_map = self._match_element_to_borehole(self._coordinates_list)
        borehole_to_groundwater_map = self._match_element_to_borehole(
            [gw.groundwater_feature_list for gw in self._groundwater_in_doc.borehole_groundwaters],
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
                self._groundwater_in_doc.borehole_groundwaters[borehole_to_groundwater_map[borehole_index]],
                layers_in_borehole_with_bb.bounding_boxes,
            )
            for borehole_index, layers_in_borehole_with_bb in enumerate(
                self._layers_with_bb_in_document.boreholes_layers_with_bb
            )
        ]

    def _extend_length_to_match_boreholes_num_pred(self) -> None:
        """Ensures that all lists have the same length by duplicating elements if necessary."""
        num_boreholes = len(self._layers_with_bb_in_document.boreholes_layers_with_bb)

        self._groundwater_in_doc.borehole_groundwaters = self._extend_list(
            self._groundwater_in_doc.borehole_groundwaters, GroundwatersInBorehole([]), num_boreholes
        )
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

        return lst[:target_length]

    def _match_element_to_borehole(
        self,
        element_list: list[FeatureOnPage | None | list[FeatureOnPage]],
    ) -> dict[int, int]:
        """Matches extracted elements to boreholes.

        This is done by minimizing the total sum of the distances from the elements to the corresponding layers.

        Args:
            element_list (list[FeatureOnPage  |  None  |  list[FeatureOnPage]]): list of element to match

        Returns:
            dict[int, int]: the dictionary containing the best mapping borehole_index -> element_index
        """
        # solve trivial case and case where the elements are None
        if len(element_list) == 1 or not element_list[0]:
            return {idx: idx for idx in range(len(element_list))}

        # Normalize input: Ensure all element in the outer list are a list (even if it's just one feature).
        # This is done in order to use the same function to treat borehole_groundwaters and the other elements
        # (elevations, coordinates), as borehole_groundwaters are a list of possibly many groundwater entries.
        elements_to_match = (
            [[elem] for elem in element_list] if isinstance(element_list[0], FeatureOnPage) else element_list
        )

        # Get the list of all borehole bounding boxes (references)
        borehole_bounding_boxes = [
            bh.bounding_boxes for bh in self._layers_with_bb_in_document.boreholes_layers_with_bb
        ]

        # Now we need to calculate the distance between each elevation and each borehole bounding box
        cost_matrix = np.zeros((len(elements_to_match), len(borehole_bounding_boxes)))

        def distance_func(feat_list: list[FeatureOnPage], bounding_boxes: list[PageBoundingBoxes]):
            """Computes the distance between a series of FeatureOnPage objects and the bounding boxes."""
            bbox = next((bbox for bbox in bounding_boxes if bbox.page == feat_list[0].page), None)
            if bbox is None:
                # the current boreholes layers don't appear on the page where the element is
                return float("inf")
            outer_rect = bbox.get_outer_rect()  # Get the outer rect of the bounding box
            # the center is the average of the extreme coordinates of each feature in the list.
            element_center = fitz.Point(0, 0)
            for feat in feat_list:
                element_center += (feat.rect.top_left + feat.rect.bottom_right) / 2
            element_center /= len(feat_list)

            dist = element_center.distance_to(outer_rect)
            return dist

        # Fill the cost matrix with distances between each element and each borehole
        for i, feat_list in enumerate(elements_to_match):
            for j, bboxes in enumerate(borehole_bounding_boxes):
                dist = distance_func(feat_list, bboxes)  # Compute the distance
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
