"""Data utils module."""

import json
from collections import Counter
from pathlib import Path

from description_classification.utils.data_loader import LayerInformations


def get_data_language_count(layer_descriptions: list[LayerInformations]) -> dict[str, int]:
    """Returns the count of sample for each language.

    Args:
        layer_descriptions (list[LayerInformations]): All the layers.

    Returns:
        dict[str,int]: the count for each language.
    """
    language_counts = dict(Counter(layer.language for layer in layer_descriptions))
    return language_counts


def get_data_class_count(layer_descriptions: list[LayerInformations]) -> dict[str, int]:
    """Returns the count of sample for each class.

    Args:
        layer_descriptions (list[LayerInformations]): All the layers.

    Returns:
        dict[str,int]: the count for each class.
    """
    class_counts = dict(Counter(layer.ground_truth_uscs_class.name for layer in layer_descriptions))
    return class_counts


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
        json.dump(output_data, f, ensure_ascii=False, indent=4)
