"""Classes for evaluating the metadata of a borehole."""

import math

from stratigraphy.evaluation.evaluation_dataclasses import (
    FileBoreholeMetadataMetrics,
    Metrics,
    OverallBoreholeMetadataMetrics,
)
from stratigraphy.metadata.coordinate_extraction import Coordinate
from stratigraphy.util.borehole_predictions import FileMetadataWithGroundTruth


class MetadataEvaluator:
    """Class for evaluating the metadata of a borehole."""

    def __init__(self, metadata_list: list[FileMetadataWithGroundTruth]) -> None:
        """Initializes the MetadataEvaluator object.

        Args:
            metadata_list (list[FileMetadataWithGroundTruth]): a list of borehole metadata predictions, with
                ground truth data associated for every borehole.
        """
        self.metadata_list = metadata_list

    def evaluate(self) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata of the file against the ground truth."""
        # Initialize the metadata correctness metrics
        metadata_metrics_list = OverallBoreholeMetadataMetrics()

        for file_data in self.metadata_list:
            # create the lists that will contain the individual score of each borehole
            elevation_metrics_list = []
            coordinate_metrics_list = []

            for borehole_data in file_data.boreholes:
                if borehole_data.ground_truth is None:
                    # when the extraction detects more borehole than there actually is in the ground truth, the wosrt
                    # predictions have no match and must be skipped for the evaluation
                    continue

                ###########################################################################################################
                ### Compute the metadata correctness for the coordinates.
                ###########################################################################################################
                extracted_coordinates = (
                    borehole_data.metadata.coordinates.feature
                    if borehole_data.metadata and borehole_data.metadata.coordinates
                    else None
                )
                ground_truth_coordinates = borehole_data.ground_truth.get("coordinates")

                coordinate_metrics = self._evaluate_coordinate(extracted_coordinates, ground_truth_coordinates)
                if borehole_data.metadata and borehole_data.metadata.coordinates:
                    borehole_data.metadata.coordinates.feature.is_correct = coordinate_metrics.tp > 0
                coordinate_metrics_list.append(coordinate_metrics)

                ############################################################################################################
                ### Compute the metadata correctness for the elevation.
                ############################################################################################################
                extracted_elevation = (
                    borehole_data.metadata.elevation.feature.elevation
                    if borehole_data.metadata and borehole_data.metadata.elevation
                    else None
                )
                ground_truth_elevation = borehole_data.ground_truth.get("reference_elevation")
                elevation_metrics = self._evaluate_elevation(extracted_elevation, ground_truth_elevation)
                if borehole_data.metadata and borehole_data.metadata.elevation:
                    borehole_data.metadata.elevation.feature.is_correct = elevation_metrics.tp > 0
                elevation_metrics_list.append(elevation_metrics)

            # perform micro-average to store the metrics of all the boreholes in the document
            metadata_metrics_list.borehole_metadata_metrics.append(
                FileBoreholeMetadataMetrics(
                    elevation_metrics=Metrics.micro_average(elevation_metrics_list),
                    coordinates_metrics=Metrics.micro_average(coordinate_metrics_list),
                    filename=file_data.filename,
                )
            )

        return metadata_metrics_list

    def _evaluate_elevation(self, extracted_elevation: float | None, ground_truth_elevation: float | None):
        """Private method used to evaluate the extracted elevation against the ground truth.

        Args:
            extracted_elevation (float | None): the extracted elevation
            ground_truth_elevation (float | None): the groundtruth elevation

        Returns:
            Metrics: the metric for this elevation.
        """
        if extracted_elevation is not None and ground_truth_elevation is not None:
            if math.isclose(extracted_elevation, ground_truth_elevation, abs_tol=0.1):
                elevation_metrics = Metrics(
                    tp=1,
                    fp=0,
                    fn=0,
                )
            else:
                elevation_metrics = Metrics(
                    tp=0,
                    fp=1,
                    fn=1,
                )
        else:
            elevation_metrics = Metrics(
                tp=0,
                fp=1 if extracted_elevation is not None else 0,
                fn=1 if ground_truth_elevation is not None else 0,
            )

        return elevation_metrics

    def _evaluate_coordinate(self, extracted_coordinates: Coordinate | None, ground_truth_coordinates: dict | None):
        """Private method used to evaluate the extracted coordinates against the ground truth.

        Args:
            extracted_coordinates (Coordinate | None): the extracted coordinates
            ground_truth_coordinates (dict | None): the groundtruth coordinates

        Returns:
            Metrics: the metric for these coordinates.
        """
        if extracted_coordinates and ground_truth_coordinates:
            if extracted_coordinates.east.coordinate_value > 2e6 and ground_truth_coordinates["E"] < 2e6:
                ground_truth_east = int(ground_truth_coordinates["E"]) + 2e6
                ground_truth_north = int(ground_truth_coordinates["N"]) + 1e6
            elif extracted_coordinates.east.coordinate_value < 2e6 and ground_truth_coordinates["E"] > 2e6:
                ground_truth_east = int(ground_truth_coordinates["E"]) - 2e6
                ground_truth_north = int(ground_truth_coordinates["N"]) - 1e6
            else:
                ground_truth_east = int(ground_truth_coordinates["E"])
                ground_truth_north = int(ground_truth_coordinates["N"])

            if (math.isclose(int(extracted_coordinates.east.coordinate_value), ground_truth_east, abs_tol=2)) and (
                math.isclose(int(extracted_coordinates.north.coordinate_value), ground_truth_north, abs_tol=2)
            ):
                coordinate_metrics = Metrics(
                    tp=1,
                    fp=0,
                    fn=0,
                )
            else:
                coordinate_metrics = Metrics(
                    tp=0,
                    fp=1,
                    fn=1,
                )
        else:
            coordinate_metrics = Metrics(
                tp=0,
                fp=1 if extracted_coordinates is not None else 0,
                fn=1 if ground_truth_coordinates is not None else 0,
            )

        return coordinate_metrics
