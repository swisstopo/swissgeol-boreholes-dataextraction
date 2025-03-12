"""This module contains classes for predictions."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeVar

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetricsCatalog
from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBoxes
from stratigraphy.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInDocument, GroundwatersInBorehole
from stratigraphy.layer.layer import LayersInBorehole, LayersInDocument
from stratigraphy.metadata.metadata import BoreholeMetadata, FileMetadata, OverallFileMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BoreholePredictions:
    """Class that hold predicted information about a single borehole."""

    borehole_index: int
    layers_in_borehole: LayersInBorehole
    file_name: str
    metadata: BoreholeMetadata
    groundwater_in_borehole: GroundwatersInBorehole
    bounding_boxes: list[BoundingBoxes]

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "borehole_index": self.borehole_index,
            "metadata": self.metadata.to_json(),
            "layers": [layer.to_json() for layer in self.layers_in_borehole.layers],
            "bounding_boxes": [bboxes.to_json() for bboxes in self.bounding_boxes],
            "groundwater": self.groundwater_in_borehole.to_json() if self.groundwater_in_borehole is not None else [],
        }

    @classmethod
    def from_json(cls, json_object, file_name) -> "BoreholePredictions":
        """Extract a BoreholePrediction object from a json dictionary.

        Args:
            json_object (dict): the json object containing the informations of the borehole
            file_name: the file name

        Returns:
            (BoreholePredictions): the extracted object
        """
        return cls(
            json_object["borehole_index"],
            LayersInBorehole.from_json(json_object["layers"]),
            file_name,
            BoreholeMetadata.from_json(json_object["metadata"]),
            GroundwatersInBorehole.from_json(json_object["groundwater"]),
            [BoundingBoxes.from_json(bbox_json) for bbox_json in json_object["bounding_boxes"]],
        )


@dataclass
class FilePredictions:
    """A class to represent predictions for a single file.

    It is responsible of grouping all the lists of elements into a single list of BoreholePrediction objects.
    """

    borehole_predictions_list: list[BoreholePredictions]
    file_metadata: FileMetadata
    file_name: str

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            **self.file_metadata.to_json(),
            "boreholes": [borehole.to_json() for borehole in self.borehole_predictions_list],
        }


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
        elevations_list: list,
        coordinates_list: list,
    ):
        """Initializes the BoreholeListBuilder with extracted borehole-related data.

        Args:
            layers_in_document (LayersInDocument): Object containing borehole layers extracted from the document.
            file_name (str): The name of the processed document.
            groundwater_in_doc (GroundwaterInDocument): Contains detected groundwater entries for boreholes.
            bounding_boxes (list[list[BoundingBoxes]]): A nested list where each inner list represents bounding boxes
                associated with a borehole.
            elevations_list (list[float]): List of terrain elevation values for detected boreholes.
            coordinates_list (list[tuple[float, float]]): List of borehole coordinates.
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
        num_boreholes = max(
            len(self._layers_in_document.boreholes_layers),
            len(self._groundwater_in_doc.borehole_groundwaters),
            len(self._bounding_boxes),
            len(self._elevations_list),
            len(self._coordinates_list),
        )

        self._layers_in_document.boreholes_layers = self._extend_list(
            self._layers_in_document.boreholes_layers, LayersInBorehole([]), num_boreholes
        )
        self._groundwater_in_doc.borehole_groundwaters = self._extend_list(
            self._groundwater_in_doc.borehole_groundwaters, GroundwatersInBorehole([]), num_boreholes
        )
        self._bounding_boxes = self._extend_list(self._bounding_boxes, [], num_boreholes)
        self._elevations_list = self._extend_list(self._elevations_list, None, num_boreholes)
        self._coordinates_list = self._extend_list(self._coordinates_list, None, num_boreholes)

    @staticmethod
    def _extend_list(lst: list[T], base_elem: T, target_length: int) -> list[T]:
        """Extends a list with deep copies of a base element until it reaches the target length.

        If the input list is empty, it initializes it with a deep copy of base_elem.
        Then, it ensures that the list reaches target_length by appending deep copies
        of the first element in the list.

        Args:
            lst (list[T]): The list to extend.
            base_elem (T): The element to use as a base for deep copying.
            target_length (int): The desired length of the list.

        Returns:
            list[T]: The extended list with deep copies of the base element.
        """
        if not lst:
            lst.append(deepcopy(base_elem))  # Ensure there's a base element if the list is empty

        base_elem = deepcopy(lst[0])  # Use the first element as the base
        while len(lst) < target_length:
            lst.append(deepcopy(base_elem))  # Append copies to match the required length

        return lst


class OverallFilePredictions:
    """A class to represent predictions for all files."""

    def __init__(self) -> None:
        """Initializes the OverallFilePredictions object."""
        self.file_predictions_list: list[FilePredictions] = []
        self.matching_pred_to_gt_boreholes = dict()  # set when evaluating layers

    def add_file_predictions(self, file_predictions: FilePredictions) -> None:
        """Add file predictions to the list of file predictions.

        Args:
            file_predictions (FilePredictions): The file predictions to add.
        """
        self.file_predictions_list.append(file_predictions)

    def get_metadata_as_dict(self) -> dict:
        """Returns the metadata of the predictions as a dictionary.

        Returns:
            dict: The metadata of the predictions as a dictionary.
        """
        return {
            "_".join([file_prediction.file_name, str(borehole_prediction.borehole_index)]): {
                "file_metadata": file_prediction.file_metadata.to_json(),
                "borehole_metadata": borehole_prediction.metadata.to_json(),
            }
            for file_prediction in self.file_predictions_list
            for borehole_prediction in file_prediction.borehole_predictions_list
        }

    def to_json(self) -> dict:
        """Converts the object to a dictionary by merging individual file predictions.

        Returns:
            dict: A dictionary representation of the object.
        """
        return {fp.file_name: fp.to_json() for fp in self.file_predictions_list}

    @classmethod
    def from_json(cls, prediction_from_file: dict) -> "OverallFilePredictions":
        """Converts a dictionary to an object.

        Args:
            prediction_from_file (dict): A dictionary representing the predictions.

        Returns:
            OverallFilePredictions: The object.
        """
        overall_file_predictions = OverallFilePredictions()
        for file_name, file_data in prediction_from_file.items():
            file_metadata = FileMetadata.from_json(file_data, file_name)

            borehole_list = [BoreholePredictions.from_json(bh_data, file_name) for bh_data in file_data["boreholes"]]

            overall_file_predictions.add_file_predictions(FilePredictions(borehole_list, file_metadata, file_name))
        return overall_file_predictions

    ############################################################################################################
    ### Evaluation methods
    ############################################################################################################

    def evaluate_metadata_extraction(self, ground_truth: GroundTruth) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            ground_truth (GroundTruth): The ground truth.

        Returns:
            OverallBoreholeMetadataMetrics: the computed metrics for the metadata.
        """
        metadata_per_file: OverallFileMetadata = OverallFileMetadata(
            [file_pred.file_name for file_pred in self.file_predictions_list],
            [[bh.metadata for bh in fp.borehole_predictions_list] for fp in self.file_predictions_list],
        )

        return MetadataEvaluator(metadata_per_file, ground_truth, self.matching_pred_to_gt_boreholes).evaluate()

    def evaluate_geology(self, ground_truth: GroundTruth) -> OverallMetricsCatalog | None:
        """Evaluate the borehole extraction predictions.

        Args:
            ground_truth (GroundTruth): The ground truth.

        Returns:
            OverallMetricsCatalog: A OverallMetricsCatalog that maps a metrics name to the corresponding
            OverallMetrics object. If no ground truth is available, None is returned.
        """
        for file_predictions in self.file_predictions_list:
            ground_truth_for_file = ground_truth.for_file(file_predictions.file_name)
            if ground_truth_for_file:
                borehole_matching_gt_to_pred = LayerEvaluator.evaluate_borehole(
                    [bh.layers_in_borehole for bh in file_predictions.borehole_predictions_list],
                    {idx: borehole_data["layers"] for idx, borehole_data in ground_truth_for_file.items()},
                )
                self.matching_pred_to_gt_boreholes[file_predictions.file_name] = borehole_matching_gt_to_pred

        languages = set(fp.file_metadata.language for fp in self.file_predictions_list)
        all_metrics = OverallMetricsCatalog(languages=languages)

        evaluator = LayerEvaluator(
            {
                fp.file_name: [bh.layers_in_borehole for bh in fp.borehole_predictions_list]
                for fp in self.file_predictions_list
            },
            ground_truth,
            self.matching_pred_to_gt_boreholes,
        )
        all_metrics.layer_metrics = evaluator.get_layer_metrics()
        all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()

        layers_in_doc_by_language = {language: dict() for language in languages}
        for file_prediction in self.file_predictions_list:
            # even if metadata can be different for boreholes in the same document, langage is the same (take index 0)
            layers_in_doc_by_language[file_prediction.file_metadata.language][file_prediction.file_name] = [
                bh.layers_in_borehole for bh in file_prediction.borehole_predictions_list
            ]

        for language, layers_in_doc_list in layers_in_doc_by_language.items():
            evaluator = LayerEvaluator(layers_in_doc_list, ground_truth, self.matching_pred_to_gt_boreholes)
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
        groundwater_entries = {
            fp.file_name: [bh.groundwater_in_borehole for bh in fp.borehole_predictions_list]
            for fp in self.file_predictions_list
        }
        overall_groundwater_metrics = GroundwaterEvaluator(
            groundwater_entries, ground_truth, self.matching_pred_to_gt_boreholes
        ).evaluate()
        all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        all_metrics.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )
        return all_metrics
