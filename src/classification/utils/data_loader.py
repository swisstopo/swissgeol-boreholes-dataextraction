"""Data loader module."""

import json
import logging
import re
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join
from pathlib import Path

from classification.utils.classification_classes import ClassificationSystem
from utils.file_utils import read_params
from utils.language_detection import detect_language_of_text

classification_params = read_params("classification_params.yml")

logger = logging.getLogger(__name__)


@dataclass
class Depths:
    """Dataclass to represent the depths of a layer."""

    start: float
    end: float


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


def load_data(
    json_path: Path, file_subset_directory: Path | None, classification_system: type[ClassificationSystem]
) -> list[LayerInformations]:
    """Loads the data from the ground truth json file.

    Args:
        json_path (Path): the ground truth json file path
        file_subset_directory (Path): Path to the directory containing the file whose names are used.
        classification_system ( type[ClassificationSystem]): The classification system used to classify the data.

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
    total_layers = skipped_count = 0
    for filename, boreholes in data.items():
        if filename_subset is not None and filename not in filename_subset:
            continue
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
        for borehole in boreholes:
            borehole_descriptions: list[LayerInformations] = []
            for layer_index, layer in enumerate(borehole["layers"]):
                if not layer["material_description"]:
                    continue
                layer_key = classification_system.get_layer_ground_truth_key()
                class_str = layer.get(layer_key, None)
                total_layers += 1
                if not class_str:
                    logger.debug(
                        f"Skipping layer: no {layer_key} in ground truth for {filename} borehole "
                        f"{borehole['borehole_index']}, layer {layer_index} with "
                        f"description {layer['material_description']}."
                    )
                    skipped_count += 1
                    continue

                ground_truth_class = classification_system.map_most_similar_class(class_str)

                material_description = resolve_reference(layer["material_description"], borehole_descriptions)
                if material_description != layer["material_description"]:
                    logger.debug(
                        f"Resolved reference: {filename} borehole {borehole['borehole_index']}, layer {layer_index}."
                    )

                borehole_descriptions.append(
                    LayerInformations(
                        filename,
                        borehole["borehole_index"],
                        layer_index,
                        language,
                        Depths(layer["depth_interval"]["start"], layer["depth_interval"]["end"]),
                        material_description,
                        classification_system,
                        ground_truth_class,
                        prediction_class=None,
                        llm_reasoning=None,
                    )
                )
            layer_descriptions.extend(borehole_descriptions)
    logger.info(
        f"Skipped {skipped_count} layers without groundtruh out of {total_layers}, "
        f"which is {skipped_count / total_layers * 100:2f}%"
    )
    return layer_descriptions


def resolve_reference(material_description: str, previous_layers: list[LayerInformations]) -> str:
    """This function identifies if a layer description is a reference to a previous layer.

    If it is, it first find the layer it refers to by looking for depths references in the material description.
    Then, it replaces the reference with the material description of the referenced layer. If no depth reference is
    found, we assume it refers to the layer immediately before it in the list of previous layers.
    If the material description does not contain any reference keywords, it returns the material description as is.

    Args:
        material_description (str): The material description to check.
        previous_layers (list[LayerInformations]): The list of previous layers to check against.

    Returns:
        str: The resolved material description, with references replaced by actual descriptions.
    """
    if not any(material_description.lower().startswith(kw) for kw in classification_params["reference_key_words"]):
        return material_description  # No reference found, return as is
    if not previous_layers:
        logger.warning("How can this layer reference a previous layer if there is no previous layer?")
        return material_description
    # Extract the depth references from the material description
    depth_str_references = re.findall(r"\d+(?:[.,]\d+)?", material_description)
    depth_str_references = depth_str_references[:2]  # We only consider the first two depth references
    clean_depth_references = [float(depth.replace(",", ".")) for depth in depth_str_references]
    clean_depth_references.sort()
    if len(clean_depth_references) != 2:
        raise NotImplementedError("The case of a single depth reference is not implemented yet. Might happen with Sp.")
    referenced_layer = next(
        (
            layer
            for layer in previous_layers[::-1]  # Iterate in reverse order to find the most recent layer
            if layer.layer_depths == Depths(clean_depth_references[0], clean_depth_references[1])
        ),
        previous_layers[-1],
    )

    # Replace the reference in the material description with the actual description of the referenced layer
    last_depth = depth_str_references[-1]
    pattern = rf"^.*?{re.escape(last_depth)}"  # Match the entire string up to the last depth reference

    # Replace it
    return re.sub(pattern, referenced_layer.material_description, material_description).strip()
