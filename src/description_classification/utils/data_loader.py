"""Data loader module."""

import json
import logging
from dataclasses import dataclass

from description_classification.utils.language_detection import detect_language
from description_classification.utils.uscs_classes import USCSClasses
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")

logger = logging.getLogger(__name__)


@dataclass
class LayerDescription:
    """_summary_."""

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    description: str


@dataclass
class LayerUSCSGroundTruth:
    """_summary_."""

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    uscs_class: USCSClasses


def load_data(json_path):
    """_summary_.

    Args:
        json_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    descriptions: list[LayerDescription] = []
    ground_truth: list[LayerUSCSGroundTruth] = []
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
                descriptions.append(
                    LayerDescription(
                        filename, borehole["borehole_index"], layer_index, language, layer["material_description"]
                    )
                )
                uscs_class = map_most_similar_uscs(layer["uscs_1"])
                if not uscs_class:
                    logger.warning(f"Unknown class: {layer['uscs_1']}, mapping it to None.")
                ground_truth.append(
                    LayerUSCSGroundTruth(filename, borehole["borehole_index"], layer_index, language, uscs_class)
                )

    return descriptions, ground_truth


def map_most_similar_uscs(uscs_str: str) -> USCSClasses | None:
    """_summary_.

    Args:
        uscs_str (str): _description_

    Returns:
        USCSClasses: _description_
    """
    for class_ in USCSClasses:
        if uscs_str.replace("-", "_") == class_.name:
            return class_
    return None
