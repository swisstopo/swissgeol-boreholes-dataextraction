"""Data loader module."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from description_classification.utils.language_detection import detect_language
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
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


def write_predictions(layers_with_predictions: list[LayerInformations], out_dir: Path):
    """Writes the predictions and ground truth data to a JSON file.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    output_file = out_dir / "uscs_class_predictions.json"

    output_data = {}

    for layer in layers_with_predictions:
        if layer.filename not in output_data:
            output_data[layer.filename] = []

        # Find an existing borehole entry
        borehole_entry = next(
            (bh for bh in output_data[layer.filename] if bh["borehole_index"] == layer.borehole_index), None
        )

        # if the borehole does not exist, create it
        if not borehole_entry:
            borehole_entry = {"borehole_index": layer.borehole_index, "layers": []}
            output_data[layer.filename].append(borehole_entry)

        borehole_entry["layers"].append(
            {
                "layer_index": layer.layer_index,
                "material_description": layer.material_description,
                "language": layer.language,
                "ground_truth_uscs_class": layer.ground_truth_uscs_class.name
                if layer.ground_truth_uscs_class
                else None,
                "prediction_uscs_class": layer.prediction_uscs_class.name if layer.prediction_uscs_class else None,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)
