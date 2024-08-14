"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    def __init__(self, path: Path):
        self.ground_truth = defaultdict(dict)

        with open(path) as in_file:
            ground_truth = json.load(in_file)
        for borehole_profile, ground_truth_item in ground_truth.items():
            layers = ground_truth_item["layers"]
            self.ground_truth[borehole_profile]["layers"] = [
                {
                    "material_description": parse_text(layer["material_description"]),
                    "depth_interval": layer["depth_interval"],
                }
                for layer in layers
                if parse_text(layer["material_description"]) != ""
            ]
            metadata = ground_truth_item["metadata"]

            # Add the groundwater to the metadata if it is present
            if "groundwater" in ground_truth_item:
                metadata["groundwater"] = ground_truth_item["groundwater"]
            self.ground_truth[borehole_profile]["metadata"] = metadata

    def for_file(self, file_name: str) -> dict:
        """Get the ground truth data for a given file.

        Args:
            file_name (str): The file name.

        Returns:
            dict: The ground truth data for the file.
        """
        if file_name in self.ground_truth:
            return self.ground_truth[file_name]
        else:
            logger.warning(f"No ground truth data found for {file_name}.")
            return {}
