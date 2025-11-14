"""This module provides functions to load, normalize, and structure description and ground truth data."""

import json
import logging
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path

from classification.utils.file_utils import read_classification_params
from swissgeol_doc_processing.utils.language_detection import detect_language_of_text

logger = logging.getLogger(__name__)

config_path = "config"
classification_params = read_classification_params("classification_params.yml", config_path)


def format_ground_truth_file(ground_truth: dict, file_subset_directory: Path | None) -> dict:
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


def get_file_language(boreholes: list) -> str:
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


def format_descriptions_file(descriptions: dict) -> dict:
    """Format the description data to match the ground truth format.

    Args:
        descriptions (dict): The description data to format.

    Returns:
        dict: A dictionary containing the formatted description data.
    """
    return {
        filename: [format_borehole(borehole, filedata["language"]) for borehole in filedata["boreholes"]]
        for filename, filedata in descriptions.items()
    }


def format_borehole(borehole: dict, language: str) -> dict:
    """Format the borehole description data to match the ground truth format.

    Args:
        borehole (dict): The borehole data to format.
        language (str): The language of the borehole data.

    Returns:
        dict: A dictionary containing the formatted borehole data.
    """
    return {
        "borehole_index": borehole["borehole_index"],
        "language": language,
        "layers": [format_layer(layer) for layer in borehole["layers"]],
    }


def format_layer(layer: dict) -> dict:
    """Format the layer description data to match the ground truth format.

    Args:
        layer (dict): The layer data to format.

    Returns:
        dict: A dictionary containing the formatted layer data.
    """
    return {
        "material_description": layer["material_description"]["text"],
        "depth_interval": format_depths(layer["depths"]),
    }


def format_depths(depths: dict) -> dict | None:
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


def resolve_reference_all_files(files: dict) -> dict:
    """Resolve layer references across all boreholes in all files.

    This function updates the 'layers' field of each borehole in-place by resolving any references
    (e.g. descriptions saying 'same as depth 2m').

    Args:
        files (dict): Dictionary of files.

    Returns:
        dict: The same input `files` dict with `layers` mutated in-place.
    """
    for filename, data in files.items():
        for borehole in data:
            before = [layer["material_description"] for layer in borehole["layers"]]
            borehole["layers"] = resolve_reference_all_layers(borehole["layers"])
            after = [layer["material_description"] for layer in borehole["layers"]]

            for b, a in zip(before, after, strict=True):
                if b != a:
                    logger.debug(f"Resolved reference: {filename} borehole {borehole['borehole_index']}, {b} -> {a}.")
    return files


def resolve_reference_all_layers(layers: list[dict]) -> list[dict]:
    """This function resolve material_description references in-place for a list of layers.

    Args:
        layers (list[dict]): The list of layers to check against.

    Returns:
        list[dict]: The list of layers with resolved references.
    """
    previous_layers = []
    for layer in layers:
        if not layer["material_description"]:
            continue
        layer["material_description"] = resolve_reference(layer["material_description"], previous_layers)
        previous_layers.append(layer)
    return layers


def resolve_reference(
    material_description: str,
    previous_layers: list[dict],
) -> str:
    """This function identifies if a layer description is a reference to a previous layer.

    If it is, it first finds the layer it refers to by looking for depths references in the material description.
    Then, it replaces the reference with the material description of the referenced layer. If no depth reference is
    found, we assume it refers to the layer immediately before it in the list of previous layers.
    If the material description does not contain any reference keywords, it returns the material description as is.

    Args:
        material_description (str): The material description to check.
        previous_layers (list[T]): The list of previous layers to check against.

    Returns:
        str: The resolved material description, with references replaced by actual descriptions.
    """
    key_words = "|".join([re.escape(kw) for kw in classification_params["reference_key_words"]])
    key_word_query = rf"^[\s\-]*(?:{key_words})\b"  # contains the capturing group
    depth_query = r"(\d+(?:[.,]\d+)?)"
    unit_query = r"(?:\s*(?:[müMN][.\s]*)+)?\b"
    total_query = (
        rf"{key_word_query}"  # match the keyword
        rf"(?:(?:[\s.]|Sp)*-?"  # open optional non-capturing group and allow for various separators (./Sp./-)
        rf"{depth_query}{unit_query}"  # match the first depth with its optional unit
        rf"(?:[\s-]*"  # open second optional non-capturing group and allow for various separators
        rf"{depth_query}{unit_query})?)?"  # match the second depth with its optional unit
    )

    match = re.match(total_query, material_description, re.IGNORECASE)
    if not match:
        return material_description
    if not previous_layers:
        logger.warning(f"Reference keyword found but no previous layer exists: '{material_description}'")
        return material_description
    if not match.group(1):  # no depth reference, fallback to the previous layer
        referenced_layer = previous_layers[-1]
    else:
        clean_depth_reference = float(match.group(1).replace(",", ".").strip())

        def match_layer(layer, depths_to_match):
            return layer["depth_interval"]["start"] == depths_to_match if layer["depth_interval"] else False

        referenced_layer = next(
            (layer for layer in reversed(previous_layers) if match_layer(layer, clean_depth_reference)),
            previous_layers[-1],  # Fallback to last previous layer
        )

    return referenced_layer["material_description"] + material_description[match.end() :]


def format_data(descriptions_path: Path, ground_truth_path: Path | None) -> tuple:
    """Load and format description and ground truth data for classification.

    Args:
        descriptions_path (Path): Path to the JSON file containing the text descriptions.
        ground_truth_path (Path | None): Path to the ground truth JSON file (or None if using a combined file).

    Returns:
        tuple:
            - descriptions (dict): Parsed and formatted descriptions data.
            - ground_truth (dict): Parsed and formatted ground truth data.
    """
    with open(descriptions_path, encoding="utf-8") as f:
        descriptions = json.load(f)
    descriptions = format_descriptions_file(descriptions)
    descriptions = resolve_reference_all_files(descriptions)

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    ground_truth = resolve_reference_all_files(ground_truth)
    return descriptions, ground_truth


def format_data_one_file(descriptions_path: Path, file_subset_directory: Path | None) -> tuple:
    """Load and format description and ground truth data for classification.

    Args:
        descriptions_path (Path): Path to the JSON file containing the text descriptions.
        file_subset_directory (Path | None): Optional directory to subset files when using single-file mode.

    Returns: data.Parsed and formatted
        tuple:
            - descriptions (dict): Parsed and formatted descriptions data.
            - ground_truth (dict): The same file, as it also contains the ground truth
    """
    logger.info("Using single-file mode: extracting texts and labels from the same file.")
    with open(descriptions_path, encoding="utf-8") as f:
        descriptions = json.load(f)
    descriptions = format_ground_truth_file(descriptions, file_subset_directory)
    descriptions = resolve_reference_all_files(descriptions)
    return descriptions, descriptions
