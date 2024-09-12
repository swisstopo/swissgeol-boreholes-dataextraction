"""Classes for evaluating the metadata of a borehole."""

import math
from typing import Any

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.evaluation.evaluation_dataclasses import (
    FileBoreholeMetadataMetrics,
    Metrics,
    OverallBoreholeMetadataMetrics,
)
from stratigraphy.metadata.metadata import BoreholeMetadataList


class MetadataEvaluator:
    """Class for evaluating the metadata of a borehole."""

    metadata_list: BoreholeMetadataList = None
    ground_truth: dict[str, Any] = None

    def __init__(self, metadata_list: BoreholeMetadataList, ground_truth_path: str):
        """Initializes the MetadataEvaluator object.

        Args:
            metadata_list (BoreholeMetadataList): The metadata to evaluate.
            ground_truth_path (str): The path to the ground truth file.
        """
        self.metadata_list = metadata_list

        # Load the ground truth data for the metadata
        self.metadata_ground_truth = GroundTruth(ground_truth_path)

    def evaluate(self) -> OverallBoreholeMetadataMetrics:
        """Evaluate the metadata of the file against the ground truth.

        Args:
            ground_truth_path (str): The path to the ground truth file.
        """
        # Initialize the metadata correctness metrics
        metadata_metrics_list = OverallBoreholeMetadataMetrics()

        for metadata in self.metadata_list.metadata_per_file:
            ###########################################################################################################
            ### Compute the metadata correctness for the coordinates.
            ###########################################################################################################
            extracted_coordinates = metadata.coordinates
            ground_truth_coordinates = (
                self.metadata_ground_truth.for_file(metadata.filename.name).get("metadata", {}).get("coordinates")
            )

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
                        feature_name="coordinates",
                    )
                else:
                    coordinate_metrics = Metrics(
                        tp=0,
                        fp=1,
                        fn=1,
                        feature_name="coordinates",
                    )
            else:
                coordinate_metrics = Metrics(
                    tp=0,
                    fp=1 if extracted_coordinates is not None else 0,
                    fn=1 if ground_truth_coordinates is not None else 0,
                    feature_name="coordinates",
                )

            ############################################################################################################
            ### Compute the metadata correctness for the elevation.
            ############################################################################################################
            extracted_elevation = None if metadata.elevation is None else metadata.elevation.elevation
            ground_truth_elevation = (
                self.metadata_ground_truth.for_file(metadata.filename.name)
                .get("metadata", {})
                .get("reference_elevation")
            )

            if extracted_elevation is not None and ground_truth_elevation is not None:
                if math.isclose(extracted_elevation, ground_truth_elevation, abs_tol=0.1):
                    elevation_metrics = Metrics(
                        tp=1,
                        fp=0,
                        fn=0,
                        feature_name="elevation",
                    )
                else:
                    elevation_metrics = Metrics(
                        tp=0,
                        fp=1,
                        fn=1,
                        feature_name="elevation",
                    )
            else:
                elevation_metrics = Metrics(
                    tp=0,
                    fp=1 if extracted_elevation is not None else 0,
                    fn=1 if ground_truth_elevation is not None else 0,
                    feature_name="elevation",
                )

            metadata_metrics_list.borehole_metadata_metrics.append(
                FileBoreholeMetadataMetrics(
                    elevation_metrics=elevation_metrics,
                    coordinates_metrics=coordinate_metrics,
                    filename=metadata.filename.name,
                )
            )

        return metadata_metrics_list
