"""This module contains classes for predictions."""

import dataclasses
import logging
from copy import deepcopy
from typing import TypeVar

from stratigraphy.benchmark.metrics import OverallMetricsCatalog
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBoxes
from stratigraphy.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInDocument, GroundwatersInBorehole
from stratigraphy.layer.layer import LayersInDocument
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
        layers_in_document: LayersInDocument,
        file_name: str,
        groundwater_in_doc: GroundwaterInDocument,
        bounding_boxes: list[list[BoundingBoxes]],
        elevations_list: list[FeatureOnPage[Elevation] | None],
        coordinates_list: list[FeatureOnPage[Coordinate] | None],
    ):
        """Initializes the BoreholeListBuilder with extracted borehole-related data.

        Args:
            layers_in_document (LayersInDocument): Object containing borehole layers extracted from the document.
            file_name (str): The name of the processed document.
            groundwater_in_doc (GroundwaterInDocument): Contains detected groundwater entries for boreholes.
            bounding_boxes (list[list[BoundingBoxes]]): A nested list where each inner list represents bounding boxes
                associated with a borehole.
            elevations_list (list[FeatureOnPage[Elevation] | None]): List of terrain elevation values for detected
                boreholes.
            coordinates_list (list[FeatureOnPage[Coordinate] | None]): List of borehole coordinates.
        """
        self._layers_in_document = layers_in_document
        self._file_name = file_name
        self._groundwater_in_doc = groundwater_in_doc
        self._bounding_boxes = bounding_boxes
        self._elevations_list = elevations_list
        self._coordinates_list = coordinates_list

    def build(self) -> list[BoreholePredictions]:
        """Creates a list of BoreholePredictions after ensuring all lists have the same length."""
        self._extend_length_to_match_boreholes_num_pred()

        return [
            BoreholePredictions(
                borehole_index,
                layers_in_borehole,
                self._file_name,
                BoreholeMetadata(elevation, coordinate),
                groundwater,
                bounding_boxes,
            )
            for borehole_index, (
                layers_in_borehole,
                groundwater,
                bounding_boxes,
                elevation,
                coordinate,
            ) in enumerate(
                zip(
                    self._layers_in_document.boreholes_layers,
                    self._groundwater_in_doc.borehole_groundwaters,
                    self._bounding_boxes,
                    self._elevations_list,
                    self._coordinates_list,
                    strict=True,
                )
            )
        ]

    def _extend_length_to_match_boreholes_num_pred(self) -> None:
        """Ensures that all lists have the same length by duplicating elements if necessary."""
        num_boreholes = len(self._layers_in_document.boreholes_layers)

        self._groundwater_in_doc.borehole_groundwaters = self._extend_list(
            self._groundwater_in_doc.borehole_groundwaters, GroundwatersInBorehole([]), num_boreholes
        )
        self._bounding_boxes = self._extend_list(self._bounding_boxes, [], num_boreholes)
        self._elevations_list = self._extend_list(self._elevations_list, None, num_boreholes)
        self._coordinates_list = self._extend_list(self._coordinates_list, None, num_boreholes)

    @staticmethod
    def _extend_list(lst: list[T], default_elem: T, target_length: int) -> list[T]:
        # deepcopy is necessary, because the is_correct attribute is already stored on this object, but the same
        # extracted value might be correct on one borehole and incorrect on another one.
        def create_new_elem():
            return deepcopy(lst[0]) if lst else default_elem

        """Extends a list with deep copies of a base element until it reaches the target length."""
        while len(lst) < target_length:
            lst.append(create_new_elem())  # Append copies to match the required length

        return lst[:target_length]


@dataclasses.dataclass
class AllBoreholePredictionsWithGroundTruth:
    """Class for evaluating all files, after individual boreholes have been match with their ground truth data."""

    predictions_list: list[FilePredictionsWithGroundTruth]

    def evaluate_metadata_extraction(self) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Returns:
            OverallBoreholeMetadataMetrics
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
