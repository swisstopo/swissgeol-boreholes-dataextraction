"""Data loader module."""

import logging
from dataclasses import dataclass
from pathlib import Path

import Levenshtein

from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_formatter import format_data, format_data_one_file
from utils.file_utils import parse_text

logger = logging.getLogger(__name__)

MATERIAL_DESCRIPTION_SIMILARITY_THRESHOLD = 0.7


@dataclass
class LayerInformation:
    """Class for each layer in the ground truth json file.

    A layer is either classified into USCS or lithology, but never both.
    """

    filename: str
    borehole_index: int
    layer_index: int
    language: str
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


def prepare_classification_data(
    descriptions_path: Path,
    ground_truth_path: Path | None,
    file_subset_directory: Path | None,
    classification_system: type[ClassificationSystem],
) -> list[LayerInformation]:
    """Load and combine material descriptions and ground truth data into a structured list.

    Args:
        descriptions_path (Path): Path to the JSON file containing the descriptions (or the file that contains
            descriptions and ground truth in single-file mode).
        ground_truth_path (Path | None): Path to the ground truth JSON file (or None if using single-file mode).
        file_subset_directory (Path | None): Directory containing a subset of filenames to use (only used if
            ground_truth_path is not provided).
        classification_system (type[ClassificationSystem]): The classification system class.

    Returns:
        list[LayerInformation]: List of structured layer information entries.
    """
    # determine if two or one files are passed, if only one it must contain both descriptions and ground truths
    single_file_mode = ground_truth_path is None

    if single_file_mode:
        descriptions, ground_truth = format_data_one_file(descriptions_path, file_subset_directory)
    else:
        descriptions, ground_truth = format_data(descriptions_path, ground_truth_path)

    layer_class_key = classification_system.get_layer_ground_truth_keys()

    layer_descriptions: list[LayerInformation] = []
    total_layers = 0
    for filename, boreholes in descriptions.items():
        if filename not in ground_truth:
            logger.warning(f"No matching ground truth for the file {filename}.")
            continue

        # Collect all layers from ground truth boreholes (safe since duplicate layers can be mixed).
        ground_truth_layers = [layer for borehole in ground_truth[filename] for layer in borehole["layers"]]

        for borehole in boreholes:
            for layer_index, layer in enumerate(borehole["layers"]):
                total_layers += 1
                if not layer["material_description"]:
                    continue

                if not single_file_mode:
                    ground_truth_match = find_matching_layer(layer, ground_truth_layers)
                    if ground_truth_match is None:
                        logger.info(f"No ground truth found for file {filename}: {layer['material_description']}.")
                        continue
                    ground_truth_layers.remove(ground_truth_match)
                else:
                    ground_truth_match = layer

                class_str = classification_system.get_class_from_entry(ground_truth_match, layer_class_key)
                if not class_str:
                    logger.debug(
                        f"Skipping layer: no {layer_class_key} in ground truth for {filename},"
                        f" layer: {layer['material_description']}."
                    )
                    continue

                ground_truth_class = classification_system.map_most_similar_class(class_str)

                layer_descriptions.append(
                    LayerInformation(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        borehole["language"],
                        layer["material_description"],
                        classification_system,
                        ground_truth_class,
                        prediction_class=None,
                        llm_reasoning=None,
                    )
                )
    skipped_count = total_layers - len(layer_descriptions)
    logger.info(
        f"Skipped {skipped_count} layers without groundtruh out of {total_layers}, "
        f"which is {skipped_count / total_layers * 100:2f}%"
    )
    return layer_descriptions
