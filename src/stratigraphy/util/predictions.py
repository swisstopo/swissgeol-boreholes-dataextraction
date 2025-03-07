"""This module contains classes for predictions."""

import logging
from copy import deepcopy

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetricsCatalog
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBoxes
from stratigraphy.evaluation.evaluation_dataclasses import OverallBoreholeMetadataMetrics
from stratigraphy.evaluation.groundwater_evaluator import GroundwaterEvaluator
from stratigraphy.evaluation.layer_evaluator import LayerEvaluator
from stratigraphy.evaluation.metadata_evaluator import MetadataEvaluator
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument, GroundwatersInBorehole
from stratigraphy.layer.layer import BoreholeLayers, Layer, LayersInDocument
from stratigraphy.metadata.metadata import BoreholeMetadata, OverallFileMetadata

logger = logging.getLogger(__name__)


class BoreholePredictions:
    """Class that hold predicted information about a single borehole."""

    def __init__(
        self,
        borehole_index: int,
        layers_in_borehole: BoreholeLayers,
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater: GroundwatersInBorehole,
        bounding_boxes: list[BoundingBoxes],
    ):
        self.borehole_index = borehole_index
        self.layers_in_borehole: BoreholeLayers = layers_in_borehole
        self.bounding_boxes: list[BoundingBoxes] = bounding_boxes
        self.file_name: str = file_name
        self.metadata: BoreholeMetadata = metadata
        self.groundwater_in_borehole: GroundwatersInBorehole = groundwater

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


class FilePredictions:
    """A class to represent predictions for a single file."""

    def __init__(
        self,
        layers_in_document: LayersInDocument,
        file_name: str,
        metadata: BoreholeMetadata,
        groundwater: GroundwaterInDocument,
        bounding_boxes: list[BoundingBoxes],
    ):
        """Instanciate a FilePrediction object.

        It is important that the order is the same for each list.For example, if layers_in_document is an orderer list
        with layers refering to borehole A, then borehole B, then metadata must also first refer to borehole A, then
        B. Or this matching could be done here but it seems less intuitive.

        Args:
            layers_in_document (LayersInDocument): list of layers of each borehole
            file_name (str): filename
            metadata (BoreholeMetadata): list TODO change to be a list
            groundwater (GroundwaterInDocument): a list of lists containing the gw detected for each bh in each doc
            bounding_boxes (list[BoundingBoxes]): _description_
        """
        self._metadata_list = metadata
        self._groundwater_in_doc = groundwater
        self._layers_in_document = layers_in_document
        self.file_name = file_name

        self.extend_lenght_to_match_boreholes_num_pred()

        self.borehole_predictions_list = [
            BoreholePredictions(
                borehole_index,
                layers_in_borehole,
                file_name,
                metadata,
                groundwater,
                bounding_boxes,
            )
            for borehole_index, (layers_in_borehole, metadata, groundwater) in enumerate(
                zip(
                    self._layers_in_document.boreholes_layers,
                    self._metadata_list,
                    self._groundwater_in_doc.borehole_groundwaters,
                    strict=True,
                )
            )
        ]

    def extend_lenght_to_match_boreholes_num_pred(self):
        """Extends the lenght of all element to match the number of borehole detected.

        If part of the pipeline is confident that there are two boreholes displayed on the pdf (for example, 2 sets of
        borehole layers) then the other element must be duplicated to be able to create 2 separate BoreholePredictions
        objects.

        """
        if not isinstance(self._metadata_list, list):
            self._metadata_list = [self._metadata_list]
        num_boreholes = max(
            len(self._layers_in_document.boreholes_layers),
            len(self._metadata_list),
            len(self._groundwater_in_doc.borehole_groundwaters),
        )

        bh_layers = self._layers_in_document.boreholes_layers
        base_bh_layer = bh_layers[0] if bh_layers else BoreholeLayers([])
        while len(bh_layers) < num_boreholes:
            bh_layers.append(deepcopy(base_bh_layer))

        md_list = self._metadata_list
        base_md = md_list[0] if md_list else BoreholeMetadata()
        while len(md_list) < num_boreholes:
            md_list.append(deepcopy(base_md))

        gw_list = self._groundwater_in_doc.borehole_groundwaters
        base_gw = gw_list[0] if gw_list else GroundwatersInBorehole([])
        while len(gw_list) < num_boreholes:
            gw_list.append(deepcopy(base_gw))

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return [borehole.to_json() for borehole in self.borehole_predictions_list]


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
            "_".join(
                [file_prediction.file_name, str(borehole_prediction.borehole_index)]
            ): borehole_prediction.metadata.to_json()
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

        # TODO still need to prop change here

        Args:
            prediction_from_file (dict): A dictionary representing the predictions.

        Returns:
            OverallFilePredictions: The object.
        """
        overall_file_predictions = OverallFilePredictions()
        for file_name, file_data in prediction_from_file.items():
            metadata = BoreholeMetadata.from_json(file_data["metadata"], file_name)

            layers = [Layer.from_json(data) for data in file_data["layers"]]
            layers_in_doc = LayersInDocument(layers=layers, filename=file_name)

            bounding_boxes = [BoundingBoxes.from_json(bboxes) for bboxes in file_data["bounding_boxes"]]

            groundwater_entries = [FeatureOnPage.from_json(entry, Groundwater) for entry in file_data["groundwater"]]
            groundwater_in_document = GroundwaterInDocument(groundwater=groundwater_entries, filename=file_name)
            overall_file_predictions.add_file_predictions(
                FilePredictions(
                    layers_in_document=layers_in_doc,
                    file_name=file_name,
                    metadata=metadata,
                    bounding_boxes=bounding_boxes,
                    groundwater=groundwater_in_document,
                )
            )
        return overall_file_predictions

    ############################################################################################################
    ### Evaluation methods
    ############################################################################################################

    def evaluate_metadata_extraction(self, ground_truth: GroundTruth) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata extraction of the predictions against the ground truth.

        Args:
            ground_truth (GroundTruth): The ground truth.

        Returns:
            OverallBoreholeMetadataMetrics
        """
        metadata_per_file: OverallFileMetadata = OverallFileMetadata(
            [
                [borehole_pred.metadata for borehole_pred in file_prediction.borehole_predictions_list]
                for file_prediction in self.file_predictions_list
            ]
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
                    file_predictions._layers_in_document,
                    {idx: borehole_data["layers"] for idx, borehole_data in ground_truth_for_file.items()},
                )
                self.matching_pred_to_gt_boreholes[file_predictions.file_name] = borehole_matching_gt_to_pred

        languages = set(metadata.language for fp in self.file_predictions_list for metadata in fp._metadata_list)
        all_metrics = OverallMetricsCatalog(languages=languages)

        evaluator = LayerEvaluator(
            [prediction._layers_in_document for prediction in self.file_predictions_list],
            ground_truth,
            self.matching_pred_to_gt_boreholes,
        )
        all_metrics.layer_metrics = evaluator.get_layer_metrics()
        all_metrics.depth_interval_metrics = evaluator.get_depth_interval_metrics()

        layers_in_doc_by_language = {language: [] for language in languages}
        for file_prediction in self.file_predictions_list:
            # even if metadata can be different for boreholes in the same document, langage is the same (take index 0)
            layers_in_doc_by_language[file_prediction._metadata_list[0].language].append(
                file_prediction._layers_in_document
            )

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
        groundwater_entries = [file_prediction._groundwater_in_doc for file_prediction in self.file_predictions_list]
        overall_groundwater_metrics = GroundwaterEvaluator(
            groundwater_entries, ground_truth, self.matching_pred_to_gt_boreholes
        ).evaluate()
        all_metrics.groundwater_metrics = overall_groundwater_metrics.groundwater_metrics_to_overall_metrics()
        all_metrics.groundwater_depth_metrics = (
            overall_groundwater_metrics.groundwater_depth_metrics_to_overall_metrics()
        )
        return all_metrics
