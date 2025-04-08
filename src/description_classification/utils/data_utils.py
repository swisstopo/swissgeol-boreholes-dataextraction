"""Data utils module."""

import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path

from description_classification.evaluation.evaluate import AllClassificationMetrics
from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses
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


def write_per_language_per_class_predictions(
    layers_with_predictions: list[LayerInformations], classification_metrics: AllClassificationMetrics, out_dir: Path
):
    """Creates json files that sumarizes the predictions.

    Creates one folder for each language and one for the global predictions. In each folder creates a file
        for each USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        classification_metrics (AllClassificationMetrics): The classification metrics computed for the predictions.
        out_dir (Path): Path to the output directory.

    """
    out_dir = out_dir / "predictions_per_class"
    for language in ["global", *classification_params["supported_language"]]:
        metrics_dict_in_language = {
            k: v for k, v in classification_metrics.per_class_all_metrics_dict.items() if k.startswith(language)
        }
        layers_in_language = [
            layer for layer in layers_with_predictions if language == "global" or layer.language == language
        ]

        write_overview(metrics_dict_in_language, layers_in_language, out_dir / language)

        # groupby ground truth first
        write_per_class_predictions(
            layers_in_language,
            groupby_ground_truth_first=True,
            metrics_dict=metrics_dict_in_language,
            out_dir=out_dir / language / "group_by_ground_truth",
        )

        # groupby prediction first
        write_per_class_predictions(
            layers_in_language,
            groupby_ground_truth_first=False,
            metrics_dict=metrics_dict_in_language,
            out_dir=out_dir / language / "group_by_prediction",
        )


def write_overview(metrics_dict: dict[str, float], layers_with_predictions: list[LayerInformations], out_dir: Path):
    """Write an overview CSV containing performance metrics per USCS class.

    Args:
        metrics_dict (dict[str, float]): Flat dictionary with metrics, e.g., 'global_CL_f1': 0.5.
        layers_with_predictions (list[LayerInformations]): List of layer predictions.
        out_dir (Path): Directory where the overview.csv will be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "overview.csv"

    # Define the columns you want in the CSV
    fieldnames = ["class", "f1", "precision", "recall", "number_ground_truth", "number_prediction"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_ in USCSClasses:
            # filter metrics dict (e.g. extract "CL_ML" for "global_CL_ML_recall")
            class_metrics_dict = {k: v for k, v in metrics_dict.items() if class_.name == "_".join(k.split("_")[1:-1])}

            num_ground_truth = sum(layer.ground_truth_uscs_class == class_ for layer in layers_with_predictions)
            num_pred = sum(layer.prediction_uscs_class == class_ for layer in layers_with_predictions)

            row = {
                "class": class_.name,
                "f1": next(v for k, v in class_metrics_dict.items() if k.endswith("f1")),
                "precision": next(v for k, v in class_metrics_dict.items() if k.endswith("precision")),
                "recall": next(v for k, v in class_metrics_dict.items() if k.endswith("recall")),
                "number_ground_truth": num_ground_truth,
                "number_prediction": num_pred,
            }
            writer.writerow(row)


def write_per_class_predictions(
    layers_with_predictions: list[LayerInformations],
    groupby_ground_truth_first: bool,
    metrics_dict: dict[str, float],
    out_dir: Path,
):
    """Creates one file per USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        groupby_ground_truth_first (bool): whether the grouping is done first by ground truth or predicted class.
        metrics_dict (dict[str, float]): the dict containing the relevent metrics.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    if groupby_ground_truth_first:
        first_key_str = "ground_truth"
        second_key_str = "prediction"
        metric_used = "recall"
    else:
        first_key_str = "prediction"
        second_key_str = "ground_truth"
        metric_used = "precision"

    def get_first_key_class(layer: LayerInformations):
        if groupby_ground_truth_first:
            return layer.ground_truth_uscs_class
        else:
            return layer.prediction_uscs_class

    def get_second_key_class(layer: LayerInformations):
        if groupby_ground_truth_first:
            return layer.prediction_uscs_class
        else:
            return layer.ground_truth_uscs_class

    # overview = {}
    for first_key_class in USCSClasses:
        output_file = out_dir / f"{first_key_str}_class_{first_key_class.name}.json"

        samples_for_first_key = [
            layer for layer in layers_with_predictions if get_first_key_class(layer) == first_key_class
        ]

        # stats about this class
        total_first_key = len(samples_for_first_key)

        first_key_class_dict = {}
        all_samples_first_key = {}

        for second_key_class in USCSClasses:
            second_key_class_dict = {}
            samples = [layer for layer in samples_for_first_key if get_second_key_class(layer) == second_key_class]
            second_key_class_dict[f"is_{first_key_str}_class"] = first_key_class == second_key_class
            second_key_class_dict[f"number_{second_key_str}"] = len(samples)
            second_key_class_dict[f"proportion_{second_key_str}"] = (
                len(samples) / total_first_key if total_first_key else 0.0
            )

            first_key_class_dict[second_key_class.name] = second_key_class_dict
            all_samples_first_key[second_key_class.name] = [
                {
                    "file_name": layer.filename,
                    "borehole_index": layer.borehole_index,
                    "layer_index": layer.layer_index,
                    "language": layer.language,
                    "material_description": layer.material_description,
                    "ground_truth_uscs_class": layer.ground_truth_uscs_class.name,
                    "prediction_uscs_class": layer.prediction_uscs_class.name,
                }
                for layer in samples
            ]

        # sort class to put class with most miss classifications first.
        first_key_class_dict = OrderedDict(
            sorted(
                first_key_class_dict.items(),
                key=lambda x: (x[1][f"is_{first_key_str}_class"], x[1][f"proportion_{second_key_str}"]),
                reverse=True,
            )
        )

        # micro_f1 = next(
        # (v for k, v in metrics_dict.items() if "f1" in k and first_key_class.name == "_".join(k.split("_")[1:-1]))
        # )

        # get the metric for the correct first key (recall if "zoom" is on groud truth, precision otherwise)
        first_key_metric_micro = next(
            (
                v
                for k, v in metrics_dict.items()
                if metric_used in k and first_key_class.name == "_".join(k.split("_")[1:-1])
            )
        )

        final_dict = OrderedDict()
        final_dict[f"number_{first_key_str}"] = total_first_key
        final_dict[f"micro_{metric_used}_for_{first_key_str}_class"] = first_key_metric_micro
        # final_dict[f"micro_f1_for_{first_key_str}_class"] = micro_f1
        final_dict.update(first_key_class_dict)
        final_dict["samples"] = all_samples_first_key

        # overview[first_key_class.name] = {
        #     f"micro_{metric_used}": first_key_metric_micro,
        #     "micro_f1": micro_f1,
        #     f"number_{first_key_str}": total_first_key,
        # }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)

    # overview = OrderedDict(
    #     sorted(overview.items(), key=lambda x: (x[1][f"micro_{metric_used}"], -x[1][f"number_{first_key_str}"]))
    # )
    # with open(out_dir / f"_overview_groupby_{first_key_str}.json", "w", encoding="utf-8") as f:
    #     json.dump(overview, f, ensure_ascii=False, indent=4)
