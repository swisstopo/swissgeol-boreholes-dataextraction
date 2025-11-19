"""Classes for evaluating the metadata of a borehole."""

import math

from extraction.evaluation.evaluation_dataclasses import (
    FileBoreholeMetadataMetrics,
    Metrics,
    OverallBoreholeMetadataMetrics,
)
from extraction.evaluation.utility import evaluate_single
from extraction.features.metadata.borehole_name_extraction import BoreholeName
from extraction.features.metadata.coordinate_extraction import Coordinate
from extraction.features.predictions.borehole_predictions import FileMetadataWithGroundTruth
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.language_filtering import normalize_spaces, remove_any_keyword

name_detection_params = read_params("name_detection_params.yml")


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
            name_metrics_list = []

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

                evaluation_result = evaluate_single(
                    extracted_coordinates, ground_truth_coordinates, self.match_coordinates
                )
                coordinate_metrics = evaluation_result.metrics
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
                evaluation_result = evaluate_single(extracted_elevation, ground_truth_elevation, self.match_elevation)
                elevation_metrics = evaluation_result.metrics
                if borehole_data.metadata and borehole_data.metadata.elevation:
                    borehole_data.metadata.elevation.feature.is_correct = elevation_metrics.tp > 0
                elevation_metrics_list.append(elevation_metrics)

                ###########################################################################################################
                ### Compute the metadata correctness for the name.
                ###########################################################################################################
                extracted_name = (
                    borehole_data.metadata.name.feature
                    if borehole_data.metadata and borehole_data.metadata.name
                    else None
                )
                ground_truth_name = borehole_data.ground_truth.get("original_name")

                evaluation_result = evaluate_single(extracted_name, ground_truth_name, self.match_name)
                name_metrics = evaluation_result.metrics
                if borehole_data.metadata and borehole_data.metadata.name:
                    borehole_data.metadata.name.feature.is_correct = name_metrics.tp > 0
                name_metrics_list.append(name_metrics)

            # perform micro-average to store the metrics of all the boreholes in the document
            metadata_metrics_list.borehole_metadata_metrics.append(
                FileBoreholeMetadataMetrics(
                    elevation_metrics=Metrics.micro_average(elevation_metrics_list),
                    coordinates_metrics=Metrics.micro_average(coordinate_metrics_list),
                    name_metrics=Metrics.micro_average(name_metrics_list),
                    filename=file_data.filename,
                )
            )

        return metadata_metrics_list

    @staticmethod
    def match_elevation(extracted_elevation: float, ground_truth_elevation: float):
        """Method used to evaluate the extracted elevation against the ground truth.

        Args:
            extracted_elevation (float): the extracted elevation
            ground_truth_elevation (float): the groundtruth elevation

        Returns:
            bool: if the extracted evaluation matches the ground truth
        """
        return math.isclose(extracted_elevation, ground_truth_elevation, abs_tol=0.01)

    @staticmethod
    def match_coordinates(extracted_coordinates: Coordinate, ground_truth_coordinates: dict):
        """Method used to evaluate the extracted coordinates against the ground truth.

        Args:
            extracted_coordinates (Coordinate): the extracted coordinates
            ground_truth_coordinates (dict): the groundtruth coordinates

        Returns:
            bool: if the extracted cooredinates match the ground truth
        """
        if extracted_coordinates.east.coordinate_value > 2e6 and ground_truth_coordinates["E"] < 2e6:
            ground_truth_east = int(ground_truth_coordinates["E"]) + 2e6
            ground_truth_north = int(ground_truth_coordinates["N"]) + 1e6
        elif extracted_coordinates.east.coordinate_value < 2e6 and ground_truth_coordinates["E"] > 2e6:
            ground_truth_east = int(ground_truth_coordinates["E"]) - 2e6
            ground_truth_north = int(ground_truth_coordinates["N"]) - 1e6
        else:
            ground_truth_east = int(ground_truth_coordinates["E"])
            ground_truth_north = int(ground_truth_coordinates["N"])

        return (math.isclose(int(extracted_coordinates.east.coordinate_value), ground_truth_east, abs_tol=2)) and (
            math.isclose(int(extracted_coordinates.north.coordinate_value), ground_truth_north, abs_tol=2)
        )

    @staticmethod
    def match_name(extracted_name: BoreholeName, ground_truth_name: dict, ignore_spaces: bool = True) -> bool:
        """Check matching between extracted and ground truth names.

        The matching is based on the filtered and normalized text. Keywords such as "bohrung" or "n°" are ignored.

        Args:
            extracted_name (BoreholeName): BoreholeName object that include detected name.
            ground_truth_name (dict): Groud truth name.
            ignore_spaces (bool, optional): Indicate if spaces are ignored during matching. Defaults to True.

        Returns:
            bool: True if texts match, False otherwise.
        """
        # Define keywords are matching names and excluded ones
        keywords_set_a = name_detection_params.get("matching_keywords", [])
        keywords_set_b = name_detection_params.get("excluded_keywords", [])
        keywords = keywords_set_a + keywords_set_b
        # Noramlize strings
        extracted_name = normalize_spaces(remove_any_keyword(extracted_name.name, keywords))
        ground_truth_name = normalize_spaces(remove_any_keyword(ground_truth_name, keywords))
        # Check if space should be ignored
        if ignore_spaces:
            extracted_name = extracted_name.replace(" ", "")
            ground_truth_name = ground_truth_name.replace(" ", "")
        # Return comparison
        return extracted_name == ground_truth_name
