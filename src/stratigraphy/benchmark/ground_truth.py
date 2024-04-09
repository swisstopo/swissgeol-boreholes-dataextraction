"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import Levenshtein
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class GroundTruthForFile:
    """Ground truth data for a single file."""

    def __init__(self, ground_truth: list):
        self.descriptions = [layer["material_description"] for layer in ground_truth]
        self.unmatched_descriptions = self.descriptions.copy()

    def is_correct(self, prediction: str) -> bool | None:
        if len(self.descriptions):
            if len(self.unmatched_descriptions):
                parsed_prediction = parse_text(prediction)
                best_match = max(
                    self.unmatched_descriptions, key=lambda ref: Levenshtein.ratio(parsed_prediction, ref)
                )
                if Levenshtein.ratio(parsed_prediction, best_match) > 0.9:
                    # ensure every ground truth entry is only matched at most once
                    self.unmatched_descriptions.remove(best_match)
                    return True
            return False
        else:
            # Return None if we don't have a ground truth for the file
            return None


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
            ]

    def for_file(self, file_name: str) -> GroundTruthForFile:
        if file_name in self.ground_truth:
            return GroundTruthForFile(self.ground_truth[file_name]["layers"])
        else:
            logger.warning(f"No ground truth data found for {file_name}.")
            return GroundTruthForFile([])
