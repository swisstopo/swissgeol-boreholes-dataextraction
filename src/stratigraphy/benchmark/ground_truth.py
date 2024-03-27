"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from pathlib import Path

import Levenshtein
from stratigraphy.util.util import parse_text

logger = logging.getLogger(__name__)


class GroundTruthForFile:
    """Ground truth data for a single file."""

    def __init__(self, descriptions: list[str]):
        """Ground truth data for a single file.

        Args:
            descriptions (list[str]): List of ground truth descriptions for the file.
        """
        self.descriptions = descriptions
        self.unmatched_descriptions = descriptions.copy()

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
        """Ground truth data for the stratigraphy benchmark.

        Args:
            path (Path): Path to the ground truth data.
        """
        with open(path) as in_file:
            ground_truth = json.load(in_file)
            self.ground_truth_descriptions = {
                filename: [
                    parse_text(entry["text"])
                    for entry in data
                    if entry["tag"] == "Material description" and parse_text(entry["text"]) != ""
                ]
                for filename, data in ground_truth.items()
            }

    def for_file(self, filename: str) -> GroundTruthForFile:
        if filename in self.ground_truth_descriptions:
            return GroundTruthForFile(self.ground_truth_descriptions[filename])
        else:
            logger.warning(f"No ground truth data found for {filename}.")
            return GroundTruthForFile([])
