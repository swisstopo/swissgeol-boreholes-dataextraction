"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    def __init__(self, path: Path):
        self.ground_truth = defaultdict(dict)

        # check if the ground truth file exists
        if not (path and os.path.exists(path)):
            logger.warning("Ground truth file not found: %s", path)
            raise FileNotFoundError("Ground truth file not found: %s", path)

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
            self.ground_truth[borehole_profile]["metadata"] = ground_truth_item["metadata"]
            self.ground_truth[borehole_profile]["groundwater"] = ground_truth_item["groundwater"]

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
