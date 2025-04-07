"""Data utils module."""

import json
from collections import Counter, OrderedDict
from pathlib import Path

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses
from sklearn.metrics import f1_score
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")


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


def write_per_language_per_class_predictions(layers_with_predictions: list[LayerInformations], out_dir: Path):
    """Creates json files that sumarizes the predictions.

    Creates one folder for each language and one for the global predictions. In each folder creates a file
        for each USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        out_dir (Path): Path to the output directory.

    """
    out_dir = out_dir / "predictions_per_ground_truth_class"
    for language in ["global", *classification_params["supported_language"]]:
        write_per_class_predictions(
            [layer for layer in layers_with_predictions if language == "global" or layer.language == language],
            out_dir=out_dir / language,
        )


def write_per_class_predictions(layers_with_predictions: list[LayerInformations], out_dir: Path):
    """Creates one file per USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    overview = {}
    for ground_truth_class in USCSClasses:
        output_file = out_dir / f"ground_truth_class_{ground_truth_class.name}.json"

        ground_truth_for_this_class = [
            layer for layer in layers_with_predictions if layer.ground_truth_uscs_class == ground_truth_class
        ]
        # stats about this ground truth class
        total_gt = len(ground_truth_for_this_class)

        gt_class_predictions = {}

        for predicted_class in USCSClasses:
            gt_class_pred_class = {}
            predictions = [
                pred for pred in ground_truth_for_this_class if pred.prediction_uscs_class == predicted_class
            ]
            gt_class_pred_class["is_ground_truth"] = ground_truth_class == predicted_class
            gt_class_pred_class["number_predicted"] = len(predictions)
            gt_class_pred_class["proportion_predicted"] = len(predictions) / total_gt if total_gt else 0.0
            gt_class_pred_class["samples"] = [
                {
                    "file_name": layer.filename,
                    "borehole_index": layer.borehole_index,
                    "layer_index": layer.layer_index,
                    "language": layer.language,
                    "material_description": layer.material_description,
                    "ground_truth_uscs_class": layer.ground_truth_uscs_class.name,
                    "prediction_uscs_class": layer.prediction_uscs_class.name,
                }
                for layer in predictions
            ]
            gt_class_predictions[predicted_class.name] = gt_class_pred_class

        # sort class to put class with most miss classifications first.
        gt_class_predictions = OrderedDict(
            sorted(gt_class_predictions.items(), key=lambda x: x[1]["proportion_predicted"], reverse=True)
        )

        final_dict = OrderedDict()
        final_dict["_number_ground_truth"] = total_gt
        micro_f1 = f1_score(
            [layer.ground_truth_uscs_class.name for layer in ground_truth_for_this_class],
            [layer.prediction_uscs_class.name for layer in ground_truth_for_this_class],
            labels=[ground_truth_class.name],
            average="micro",
            zero_division=0,
        )
        final_dict["_micro_f1_for_ground_truth_class"] = micro_f1
        final_dict.update(gt_class_predictions)

        overview[ground_truth_class.name] = {"micro_f1": micro_f1, "number_ground_truth": total_gt}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)

    overview = OrderedDict(sorted(overview.items(), key=lambda x: (x[1]["micro_f1"], -x[1]["number_ground_truth"])))
    with open(out_dir / "_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=4)
