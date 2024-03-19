"""Ground truth data classes for the stratigraphy benchmark."""

import json
import re

import Levenshtein


class GroundTruthForFile:
    """Ground truth data for a single file."""

    def __init__(self, descriptions: list[str]):
        """Ground truth data for a single file.

        Args:
            descriptions (list[str]): List of ground truth descriptions for the file.
        """
        self.descriptions = descriptions
        self.unmatched_descriptions = descriptions.copy()

    def is_correct(self, prediction: str) -> bool:
        parsed_prediction = GroundTruth.parse(prediction)
        if len(self.unmatched_descriptions):
            best_match = max(self.unmatched_descriptions, key=lambda ref: Levenshtein.ratio(parsed_prediction, ref))
            if Levenshtein.ratio(parsed_prediction, best_match) > 0.9:
                # ensure every ground truth entry is only matched at most once
                self.unmatched_descriptions.remove(best_match)
                return True
            else:
                return False


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    not_alphanum = re.compile(r"[^\w\d]", re.U)

    def __init__(self, path: str):
        """Ground truth data for the stratigraphy benchmark.

        Args:
            path (str): Path to the ground truth data.
        """
        with open(path) as in_file:
            ground_truth = json.load(in_file)
            self.ground_truth_descriptions = {
                filename: [
                    GroundTruth.parse(entry["text"])
                    for entry in data
                    if entry["tag"] == "Material description" and GroundTruth.parse(entry["text"]) != ""
                ]
                for filename, data in ground_truth.items()
            }

    @staticmethod
    def parse(text: str) -> str:
        return GroundTruth.not_alphanum.sub("", text).lower()

    def for_file(self, filename: str) -> GroundTruthForFile:
        if filename in self.ground_truth_descriptions:
            return GroundTruthForFile(self.ground_truth_descriptions[filename])
        else:
            return GroundTruthForFile([])
