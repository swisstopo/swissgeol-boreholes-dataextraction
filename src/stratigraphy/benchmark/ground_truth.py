"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class GroundTruthForFile:
    """Ground truth data for a single file.

    Evaluates the prediction against the ground truth.
    """

    def __init__(self, ground_truth_layers: list):
        self.layers = ground_truth_layers
        self.num_layers = len(ground_truth_layers)


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    def __init__(self, path: Path):
        self.ground_truth = defaultdict(dict)

        with open(path) as in_file:
            ground_truth = json.load(in_file)
        for borehole_profile, layers in ground_truth.items():
            layers = layers["layers"]
            self.ground_truth[borehole_profile]["layers"] = [
                {
                    "material_description": parse_text(layer["material_description"]),
                    "depth_interval": layer["depth_interval"],
                }
                for layer in layers
                if parse_text(layer["material_description"]) != ""
            ]

    def for_file(self, file_name: str) -> GroundTruthForFile:
        if file_name in self.ground_truth:
            return GroundTruthForFile(self.ground_truth[file_name]["layers"])
        else:
            logger.warning(f"No ground truth data found for {file_name}.")
            return GroundTruthForFile([])
