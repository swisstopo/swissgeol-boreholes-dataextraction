"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from swissgeol_doc_processing.utils.file_utils import parse_text

logger = logging.getLogger(__name__)


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    def __init__(self, path: Path) -> None:
        """Instanciate the GroundTruth object.

        Args:
            path (Path): the path to the Ground truth file
        """
        self.ground_truth = defaultdict(lambda: defaultdict(dict))

        # Load the ground truth data
        with open(path, encoding="utf-8") as in_file:
            ground_truth = json.load(in_file)

        # Parse the ground truth data
        for file_name, ground_truth_item in ground_truth.items():
            for borehole_data in ground_truth_item:
                layers = borehole_data["layers"]
                borehole_index = borehole_data["borehole_index"]
                self.ground_truth[file_name][borehole_index]["layers"] = [
                    {
                        "material_description": parse_text(layer["material_description"]),
                        "depth_interval": layer["depth_interval"],
                    }
                    for layer in layers
                ]
                self.ground_truth[file_name][borehole_index]["metadata"] = borehole_data["metadata"]
                self.ground_truth[file_name][borehole_index]["groundwater"] = borehole_data["groundwater"]

    def for_file(self, file_name: str) -> dict:
        """Get the ground truth data for a given file.

        Args:
            file_name (str): The file name.

        Returns:
            dict: The ground truth data for the file.
        """
        if file_name in self.ground_truth:
            return self.ground_truth[file_name]

        logger.warning("No ground truth data found for %s.", file_name)
        return {}
