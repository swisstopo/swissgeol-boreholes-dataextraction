"""Data loader module."""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import TypeVar

import Levenshtein
from classification.utils.classification_classes import ClassificationSystem
from utils.file_utils import parse_text, read_params
from utils.language_detection import detect_language_of_text

classification_params = read_params("classification_params.yml")

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.9


@dataclass
class Depths:
    """Dataclass to represent the depths of a layer."""

    start: float
    end: float

    @classmethod
    def from_depth(cls, depths: dict) -> "Depths":
        return cls(depths["start"], depths["end"]) if depths else None


@dataclass
class LayerInformations:
    """Class for each layer in the ground truth json file.

    A layer is either classified into USCS or lithology, but never both.
    """

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    layer_depths: Depths
    material_description: str
    class_system: type[ClassificationSystem]  # note: class_system is the class, and not an instance of the class
    ground_truth_class: None | ClassificationSystem.EnumMember
    prediction_class: None | ClassificationSystem.EnumMember  # dynamically set
    llm_reasoning: None | str  # dynamically set,


def is_valid_depth_interval(layer_depths, start: float, end: float) -> bool:
    """Validate if self and the depth interval start-end match.

    Args:
        layer_depths (dict): The depths of the current layer.
        start (float): The start value of the ground truth interval.
        end (float): The end value of the ground truth interval.

    Returns:
        bool: True if the depth intervals match, False otherwise.
    """
    if layer_depths is None:
        return False

    layer_start = layer_depths["start"]
    layer_end = layer_depths["end"]
    if layer_start is None:
        return (start == 0) and (end == layer_end)

    if (layer_start is not None) and (layer_end is not None):
        return start == layer_start and end == layer_end

    return False


def find_matching_layer(layer, unmatched_ground_truth_layers) -> None | dict:
    """Find a matching layer in the ground truth layers.

    Args:
        layer (dict): The layer to match.
        unmatched_ground_truth_layers (list): The list of unmatched ground truth layers.

    Returns:
        None | dict: The matching layer from the ground truth layers or None if no match is found.
    """
    parsed_text = parse_text(layer["material_description"])
    possible_matches = [
        ground_truth_layer
        for ground_truth_layer in unmatched_ground_truth_layers
        if Levenshtein.ratio(parsed_text, parse_text(ground_truth_layer["material_description"]))
        > MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD
    ]

    if not possible_matches:
        return None

    for possible_match in possible_matches:
        start = possible_match["depth_interval"]["start"]
        end = possible_match["depth_interval"]["end"]

        if is_valid_depth_interval(layer["depth_interval"], start, end):
            return possible_match

    return max(possible_matches, key=lambda x: Levenshtein.ratio(parsed_text, parse_text(x["material_description"])))


def match_data(predictions_path, ground_truth_path, classification_system: type[ClassificationSystem]) -> dict:
    """Match the predictions with the ground truth data.

    Need to think about it to stay DRY.

    Args:
        predictions_path (Path): Path to the predictions json file.
        ground_truth_path (Path): Path to the ground truth json file.
        classification_system (type[ClassificationSystem]): The classification system used to classify the data.

    Returns:
        dict: A dictionary containing the matched data.
    """
    layer_key = classification_system.get_layer_ground_truth_key()

    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    layer_descriptions = []

    for filename, file_data in predictions.items():
        if filename not in ground_truth:
            logger.warning(f"No matching ground truth for the file {filename}.")
            continue

        # taking all the layers at once should not pose any problems
        ground_truth_layers = [layer for borehole in ground_truth[filename] for layer in borehole["layers"]]
        for borehole in file_data["boreholes"]:
            for layer_index, layer in enumerate(borehole["layers"]):
                ground_truth_match = find_matching_layer(layer, ground_truth_layers)
                if ground_truth_match is None:
                    logger.info(
                        f"No match found for layer {layer_index}: {layer['material_description']['text']} : Skipping."
                    )
                    continue
                ground_truth_layers.remove(ground_truth_match)
                class_str = ground_truth_match.get(layer_key, None)
                if not class_str:
                    logger.info("skip, because no ground truth.")
                ground_truth_class = classification_system.map_most_similar_class(class_str)
                layer_descriptions.append(
                    LayerInformations(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        file_data["language"],
                        layer["material_description"]["text"],
                        classification_system,
                        ground_truth_class,
                        prediction_class=None,
                        llm_reasoning=None,
                    )
                )

        return layer_descriptions


def format_ground_truth(ground_truth, file_subset_directory):
    """Format the ground truth data by identifiying the language and restricting the files to the subset.

    Args:
        ground_truth (dict): The ground truth data to format.
        file_subset_directory (Path | None): Path to the directory containing the file whose names are used.

    Returns:
        dict: A dictionary containing the formatted ground truth data.
    """
    logger.info("Formatting ground truth data.")
    filename_subset = None
    if file_subset_directory is not None:
        logger.info(f"Using files from subset directory: {file_subset_directory}")
        filename_subset = {f for f in listdir(file_subset_directory) if isfile(join(file_subset_directory, f))}
    return {
        filename: [
            {
                "borehole_index": borehole["borehole_index"],
                "language": get_file_language(boreholes),
                "layers": borehole["layers"],
            }
            for borehole in boreholes
        ]
        for filename, boreholes in ground_truth.items()
        if filename_subset is None or filename in filename_subset
    }


def get_file_language(boreholes: list):
    """Detect the language of the material descriptions in the boreholes.

    Args:
        boreholes (list): List of boreholes containing layers with material descriptions.

    Returns:
        str: The detected language of the material descriptions.
    """
    all_text = " ".join(
        [
            lay["material_description"]
            for bh in boreholes
            for lay in bh["layers"]
            if lay["material_description"] is not None
        ]
    )
    language = detect_language_of_text(
        all_text, classification_params["default_language"], classification_params["supported_language"]
    )
    return language


def format_predictions(predictions):
    """Format the predictions data to match the ground truth format.

    Args:
        predictions (dict): The predictions data to format.

    Returns:
        dict: A dictionary containing the formatted predictions data.
    """
    return {
        filename: [format_borehole(borehole, filedata["language"]) for borehole in filedata["boreholes"]]
        for filename, filedata in predictions.items()
    }


def format_borehole(borehole, language):
    """Format the borehole predictions data to match the ground truth format.

    Args:
        borehole (dict): The borehole data to format.
        language (str): The language of the borehole data.

    Returns:
        dict: A dictionary containing the formatted predictions data.
    """
    return {
        "borehole_index": borehole["borehole_index"],
        "language": language,
        "layers": [format_layer(layer) for layer in borehole["layers"]],
    }


def format_layer(layer):
    """Format the layer predictions data to match the ground truth format.

    Args:
        layer (dict): The layer data to format.

    Returns:
        dict: A dictionary containing the formatted layer data.
    """
    return {
        "material_description": layer["material_description"]["text"],
        "depth_interval": format_depths(layer["depths"]),
    }


def format_depths(depths):
    """Format the depths data to match the ground truth format.

    Args:
        depths (dict): The depths data to format.

    Returns:
        dict: A dictionary containing the formatted depths data.
    """
    return (
        {
            "start": depths["start"]["value"] if depths["start"] else None,
            "end": depths["end"]["value"] if depths["end"] else None,
        }
        if depths
        else None
    )


def parse_data_paths(predictions_path, ground_truth_path, file_subset_directory):
    """Parse the data paths and load the predictions and ground truth data.

    Args:
        predictions_path (Path): Path to the predictions json file.
        ground_truth_path (Path | None): Path to the ground truth json file.
        file_subset_directory (Path | None): Path to the directory containing the file whose names are used.

    Returns:
        tuple: A tuple containing the predictions and ground truth data.
    """
    if ground_truth_path is None:
        with open(predictions_path, encoding="utf-8") as f:
            predictions = json.load(f)
        predictions = format_ground_truth(predictions, file_subset_directory)
        ground_truth = predictions  # carefull, reference

    else:
        with open(predictions_path, encoding="utf-8") as f:
            predictions = json.load(f)
        predictions = format_predictions(predictions)

        with open(ground_truth_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
    return predictions, ground_truth


def load_data(
    prediction_path: Path,
    ground_truth_path: Path | None,
    file_subset_directory: Path | None,
    classification_system: type[ClassificationSystem],
) -> list[LayerInformations]:
    """Loads the data from the ground truth json file.

    Args:
        prediction_path (Path): the ground truth json file path
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        classification_system ( type[ClassificationSystem]): The classification system used to classify the data.
        ground_truth_path (Path): Path to the ground truth file.

    Returns:
        list[LayerInformations]: the data formated as a list of LayerInformations objects
    """
    # if ground_truth_path is not None:
    #     return match_data(prediction_path, ground_truth_path, classification_system)

    predictions, ground_truth = parse_data_paths(prediction_path, ground_truth_path, file_subset_directory)

    layer_key = classification_system.get_layer_ground_truth_key()

    layer_descriptions: list[LayerInformations] = []
    for filename, boreholes in predictions.items():
        if filename not in ground_truth:
            logger.warning(f"No matching ground truth for the file {filename}.")
            continue

        ground_truth_layers = [
            layer
            for borehole in ground_truth[filename]
            for layer in resolve_reference_ground_truth_all_layers(borehole["layers"])
        ]

        for borehole in boreholes:
            borehole_descriptions: list[LayerInformations] = []
            for layer_index, layer in enumerate(borehole["layers"]):
                if not layer["material_description"]:
                    continue

                if layer_key not in layer:
                    old = layer["material_description"]
                    layer["material_description"] = resolve_reference_predictions(
                        layer["material_description"], borehole_descriptions
                    )
                    if old != layer["material_description"]:
                        logger.debug(
                            f"Resolved reference: {filename} borehole"
                            f" {borehole['borehole_index']}, layer {layer_index}."
                        )

                    ground_truth_match = find_matching_layer(layer, ground_truth_layers)
                    if ground_truth_match is None:
                        logger.info(
                            f"No match found for layer {layer_index}: {layer['material_description']} : Skipping."
                        )
                        continue
                    ground_truth_layers.remove(ground_truth_match)
                else:
                    ground_truth_match = layer

                class_str = ground_truth_match.get(layer_key, None)
                if not class_str:
                    logger.debug(
                        f"Skipping layer: no {layer_key} in ground truth for {filename} borehole "
                        f"{borehole['borehole_index']}, layer {layer_index} with "
                        f"description {layer['material_description']}."
                    )
                    continue

                ground_truth_class = classification_system.map_most_similar_class(class_str)

                borehole_descriptions.append(
                    LayerInformations(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        borehole["language"],
                        Depths.from_depth(layer["depth_interval"]),
                        layer["material_description"],
                        classification_system,
                        ground_truth_class,
                        prediction_class=None,
                        llm_reasoning=None,
                    )
                )
            layer_descriptions.extend(borehole_descriptions)
    return layer_descriptions


def resolve_reference_ground_truth_all_layers(layers):
    """This function identifies if a layer description is a reference to a previous layer.

    Args:
        layers (list[dict]): The list of layers to check against.

    Returns:
        list[dict]: The list of layers with resolved references.
    """

    def match_layer(layer, depths_to_match):
        if len(depths_to_match) == 1:
            return layer["depths_interval"]["start"] == depths_to_match[0]
        elif len(depths_to_match) == 2:
            return (
                layer["depth_interval"]["start"] == depths_to_match[0]
                and layer["depth_interval"]["end"] == depths_to_match[1]
            )
        return False  # No reference found - fallback to previous layer

    def get_material_description(layer):
        return layer["material_description"]

    previous_layers = []
    for layer in layers:
        if not layer["material_description"]:
            continue
        material_description = resolve_reference(
            layer["material_description"], previous_layers, match_layer, get_material_description
        )
        layer["material_description"] = material_description
        previous_layers.append(layer)
    return previous_layers


def resolve_reference_predictions(material_description: str, previous_layers: list[LayerInformations]):
    """This function identifies if a layer description is a reference to a previous layer.

    Args:
        material_description (str): The material description to check.
        previous_layers (list[LayerInformations]): The list of previous layers to check against.

    Returns:
        str: The resolved material description, with references replaced by actual descriptions.
    """

    def match_layer(layer, depths_to_match):
        if len(depths_to_match) == 1:
            return layer.layer_depths.start == depths_to_match[0]
        elif len(depths_to_match) == 2:
            return layer.layer_depths == Depths(depths_to_match[0], depths_to_match[1])
        return False  # No reference found - fallback to previous layer

    def get_material_description(layer):
        return layer.material_description

    return resolve_reference(material_description, previous_layers, match_layer, get_material_description)


T = TypeVar("T", LayerInformations, dict)


def resolve_reference(
    material_description: str,
    previous_layers: list[T],
    match_layer: Callable[[T], bool],
    get_material_description: Callable[[T], str],
) -> str:
    """This function identifies if a layer description is a reference to a previous layer.

    If it is, it first finds the layer it refers to by looking for depths references in the material description.
    Then, it replaces the reference with the material description of the referenced layer. If no depth reference is
    found, we assume it refers to the layer immediately before it in the list of previous layers.
    If the material description does not contain any reference keywords, it returns the material description as is.

    Args:
        material_description (str): The material description to check.
        previous_layers (list[T]): The list of previous layers to check against.
        match_layer (Callable[[T], bool]): A function that checks if a layer matches the given depth references.
        get_material_description (Callable[[T], str]): A function that retrieves the material description from a layer.

    Returns:
        str: The resolved material description, with references replaced by actual descriptions.
    """
    matched_kw = next(
        (
            kw
            for kw in classification_params["reference_key_words"]
            if material_description.lower().startswith(kw.lower())
        ),
        None,
    )

    if not matched_kw:
        return material_description  # No reference found, return as is
    if not previous_layers:
        logger.warning("How can this layer reference a previous layer if there is no previous layer?")
        return material_description
    # Extract the depth references from the material description
    depth_str_references = re.findall(r"\d+(?:[.,]\d+)?", material_description)
    depth_str_references = depth_str_references[:2]  # There should be at most two depth references (start and end)

    # clean the references and find a match
    clean_depth_references = [float(depth.replace(",", ".")) for depth in depth_str_references]
    clean_depth_references.sort()

    referenced_layer = next(
        (layer for layer in reversed(previous_layers) if match_layer(layer, clean_depth_references)),
        previous_layers[-1],  # Fallback to last previous layer
    )

    # Replace the reference in the material description with the actual description of the referenced layer
    pattern = rf"^.*?{re.escape(depth_str_references[-1]) if depth_str_references else re.escape(matched_kw)}"

    # Replace it
    return re.sub(pattern, get_material_description(referenced_layer), material_description).strip()
