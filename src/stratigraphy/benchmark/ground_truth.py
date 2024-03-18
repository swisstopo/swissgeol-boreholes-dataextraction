import json
import Levenshtein
import re


class GroundTruthForFile:
    def __init__(self, descriptions: list[str]):
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
    not_alphanum = re.compile(r'[^\w\d]', re.U)

    def __init__(self, path: str):
        with open(path, 'r') as in_file:
            ground_truth = json.load(in_file)
            self.ground_truth_descriptions = {
                filename: [
                    GroundTruth.parse(entry["text"])
                    for entry in data
                    if entry["tag"] == "Material description" and GroundTruth.parse(entry["text"]) != ""]
                for filename, data in ground_truth.items()
            }

    @staticmethod
    def parse(text: str) -> str:
        return GroundTruth.not_alphanum.sub('', text).lower()

    def for_file(self, filename: str) -> GroundTruthForFile:
        if filename in self.ground_truth_descriptions:
            return GroundTruthForFile(self.ground_truth_descriptions[filename])
        else:
            return GroundTruthForFile([])
