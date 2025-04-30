"""Data loader module."""

import json
import logging
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join
from pathlib import Path

from classification.utils.classification_classes import ClassEnum, map_most_similar_class
from utils.file_utils import read_params
from utils.language_detection import detect_language_of_text

classification_params = read_params("classification_params.yml")

logger = logging.getLogger(__name__)


@dataclass
class LayerInformations:
    """Class for each layer in the ground truth json file."""

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    material_description: str
    data_type: str  # a layer is either classified into USCS or lithology, but never both
    ground_truth_class: None | ClassEnum
    prediction_class: None | ClassEnum  # dynamically set
    llm_reasoning: None | str  # dynamically set,


def load_data(json_path: Path, file_subset_directory: Path | None, data_type: str) -> list[LayerInformations]:
    """Loads the data from the ground truth json file.

    Args:
        json_path (Path): the ground truth json file path
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        data_type (str): Type of data that need to be classify

    Returns:
        list[LayerInformations]: the data formated as a list of LayerInformations objects
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if file_subset_directory is None:
        logger.info(f"Using all filenames in: {json_path}")
        filename_subset = None
    else:
        logger.info(f"Using files from subset directory: {file_subset_directory}")
        filename_subset = {f for f in listdir(file_subset_directory) if isfile(join(file_subset_directory, f))}
    layer_descriptions: list[LayerInformations] = []
    for filename, boreholes in data.items():
        if filename_subset is not None and filename not in filename_subset:
            continue
        all_text = " ".join([lay["material_description"] for bh in boreholes for lay in bh["layers"]])
        language = detect_language_of_text(
            all_text, classification_params["default_language"], classification_params["supported_language"]
        )
        for borehole in boreholes:
            for layer_index, layer in enumerate(borehole["layers"]):
                key = "uscs_1" if data_type == "uscs" else "lithology"
                class_str = layer.get(key, None)
                if not class_str:
                    logger.debug(
                        f"Skipping layer: no {data_type.upper()} ground truth for {filename} borehole "
                        f"{borehole['borehole_index']}, layer {layer_index} with "
                        f"description {layer['material_description']}."
                    )
                    continue

                ground_truth_class = map_most_similar_class(class_str, data_type)

                layer_descriptions.append(
                    LayerInformations(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        language,
                        layer["material_description"],
                        data_type,
                        ground_truth_class,
                        prediction_class=None,
                        llm_reasoning=None,
                    )
                )
    return layer_descriptions
