"""Prediction classes for stratigraphy."""

import abc
from dataclasses import dataclass

from stratigraphy.groundwater.groundwater_extraction import GroundwaterInformationOnPage
from stratigraphy.metadata.metadata import StratigraphyMetadata


@dataclass
class Layer(metaclass=abc.ABCMeta):
    """Layer class definition."""

    # TODO: Add layer properties


@dataclass
class DepthMaterialColumnPair(metaclass=abc.ABCMeta):
    """Depth Material Column Pair class definition."""

    # TODO: Add depth material column pair properties


@dataclass
class FilePredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy from a single file."""

    groundwater: list[GroundwaterInformationOnPage] | None = None
    layers: list[Layer] = None
    depths_materials_column_pairs: list[DepthMaterialColumnPair] = None
    filename: str = None

    def __init__(self, filename: str):
        """Initializes the class.

        Args:
            filename (str): The filename.
        """
        self.filename = filename

    def set_groundwater(self, groundwater: list[GroundwaterInformationOnPage]):
        """Sets the groundwater.

        Args:
            groundwater (GroundwaterInformationOnPage): The groundwater.
        """
        self.groundwater = groundwater


@dataclass
class ExtractedFileInformation(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    single_file_predictions: FilePredictions = None
    metadata: StratigraphyMetadata = None
    filename: str = None


@dataclass
class StratigraphyPredictions(metaclass=abc.ABCMeta):
    """Prediction data for stratigraphy."""

    extracted_file_information: list[ExtractedFileInformation] = None

    def __init__(self, extracted_file_information: list[ExtractedFileInformation]):
        """Initializes the class.

        Args:
            extracted_file_information (List[ExtractedFileInformation]): The extracted file information.
        """
        self.extracted_file_information = extracted_file_information
