"""Prediction classes for stratigraphy."""

import abc
from dataclasses import dataclass

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.groundwater.groundwater_extraction import GroundwaterInformationOnPage
from stratigraphy.layer.layer import LayerPrediction
from stratigraphy.metadata.metadata import BoreholeMetadata


@dataclass
class DepthMaterialColumnPair(metaclass=abc.ABCMeta):
    """Depth Material Column Pair class definition."""

    # TODO: Add depth material column pair properties


@dataclass
class FilePredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy from a single file."""

    groundwater: list[GroundwaterInformationOnPage] | None = None
    layers: list[LayerPrediction] = None
    depths_materials_column_pairs: list[DepthMaterialColumnPair] = None
    filename: str = None


@dataclass
class ExtractedFileInformation(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    single_file_predictions: FilePredictions = None
    metadata: BoreholeMetadata = None
    filename: str = None


@dataclass
class StratigraphyPredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    extracted_file_information: list[ExtractedFileInformation] = []

    def get_groundtruth(self, ground_truth_path: str) -> None:
        """Reads the ground truth from a file.

        Args:
            ground_truth_path (str): The path to the ground truth file.
        """
        ground_truth = GroundTruth(ground_truth_path)

        # TODO: Add code to get the ground truth from the ground truth object

    def add_extracted_file_information(self, extracted_file_information: ExtractedFileInformation):
        """Adds extracted file information.

        Args:
            extracted_file_information (ExtractedFileInformation): The extracted file information.
        """
        self.extracted_file_information.append(extracted_file_information)
