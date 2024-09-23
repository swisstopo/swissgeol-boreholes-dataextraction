"""Classes for evaluating the groundwater levels of a borehole."""

from typing import Any

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.groundwater.groundwater_extraction import GroundwaterList
from stratigraphy.metadata.metadata import BoreholeMetadataList


class GroundwaterEvaluator:
    """Class for evaluating the extracted groundwater information of a borehole."""

    groundwater_list: GroundwaterList = None
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
