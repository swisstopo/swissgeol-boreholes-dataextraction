"""Data loader module."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from description_classification.utils.language_detection import detect_language
from description_classification.utils.uscs_classes import USCSClasses
from stratigraphy.util.util import read_params

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
    ground_truth_uscs_class: None | USCSClasses
    prediction_uscs_class: None | USCSClasses  # dynamically set


def load_data(json_path: Path) -> list[LayerInformations]:
    """Loads the data from the ground truth json file.

    Args:
        json_path (Path): the ground truth json file path

    Returns:
        list[LayerInformations]: the data formated as a list of LayerInformations objects
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    layer_descriptions: list[LayerInformations] = []
    for filename, boreholes in data.items():
        all_text = " ".join([lay["material_description"] for bh in boreholes for lay in bh["layers"]])
        language = detect_language(
            all_text, classification_params["default_language"], classification_params["supported_language"]
        )
        for borehole in boreholes:
            for layer_index, layer in enumerate(borehole["layers"]):
                if not layer["uscs_1"]:
                    logger.debug(
                        f"Skippping layer: no ground truth for {filename}, borehole {borehole['borehole_index']}, "
                        f"layer {layer_index} with description {layer['material_description']}."
                    )
                    continue
                uscs_class = map_most_similar_uscs(layer["uscs_1"])
                if not uscs_class:
                    logger.warning(f"Unknown class: {layer['uscs_1']}, mapping it to None.")

                layer_descriptions.append(
                    LayerInformations(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        language,
                        layer["material_description"],
                        uscs_class,
                        None,
                    )
                )

    return layer_descriptions


def map_most_similar_uscs(uscs_str: str) -> USCSClasses | None:
    """Maps the ground truth string to one of the USCSClasses.

    Args:
        uscs_str (str): the ground truth string

    Returns:
        USCSClasses: the matching class
    """
    for class_ in USCSClasses:
        if uscs_str.replace("-", "_") == class_.name:
            return class_
    return None
