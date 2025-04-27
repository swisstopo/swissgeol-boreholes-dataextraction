"""Data utils module."""

import csv
import json
import os
import shutil
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from classification.evaluation.evaluate import AllClassificationMetrics
from classification.utils.data_loader import LayerInformations
from classification.utils.uscs_classes import USCSClasses
from utils.file_utils import read_params

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


def write_predictions(
    layers_with_predictions: list[LayerInformations], out_dir: Path, out_path: str = "uscs_class_predictions.json"
):
    """Writes the predictions and ground truth data to a JSON file.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        out_dir (Path): Path to the output directory.
        out_path (str): Name of the output file (default: "uscs_class_predictions.json").
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    output_file = out_dir / out_path

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
                if layer.ground_truth_uscs_class is not None
                else None,
                "prediction_uscs_class": layer.prediction_uscs_class.name
                if layer.prediction_uscs_class is not None
                else None,
                "llm_reasoning": layer.llm_reasoning 
                if layer.llm_reasoning is not None 
                else None,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def write_api_failures(api_failures: list, output_directory: Path, filename: str = "api_failure.json") -> None:
    """Write API call failures to a JSON file.

    Args:
        api_failures: List of dictionaries containing API failure information
        output_directory: Directory where the file should be saved
        filename: Name of the output file (default: "api_failure.json")
    """
    if not api_failures:
        return

    os.makedirs(output_directory, exist_ok=True)
    failures_path = output_directory / filename

    existing_failures = []
    if failures_path.exists():
        try:
            with open(failures_path) as f:
                existing_failures = json.load(f)
        except json.JSONDecodeError:
            # Overwrite the file if it isn't a valid JSON
            print(f"Warning: Existing file {failures_path} contained invalid JSON and will be overwritten")

    all_failures = existing_failures + api_failures

    with open(failures_path, "w") as f:
        json.dump(all_failures, f, indent=2)

    print(f"Recorded {len(api_failures)} failed API calls to {failures_path}, total records: {len(all_failures)}")


@dataclass
class KeyClassConfig:
    """Contains the appropriate functions depending on whether we are grouping by ground truth or prediction.

    Attributes:
        get_first_key_class: function to extract the first key class
        get_second_key_class: function to extract the second key class
        first_key_str: string label for the first key
        second_key_str: string label for the second key
        metric_used: the metric name ("recall" or "precision")
    """

    get_first_key_class: Callable[[LayerInformations], USCSClasses]
    get_second_key_class: Callable[[LayerInformations], USCSClasses]
    first_key_str: str
    second_key_str: str
    metric_used: str


def write_per_language_per_class_predictions(
    layers_with_predictions: list[LayerInformations], classification_metrics: AllClassificationMetrics, out_dir: Path
):
    """Creates json files that sumarizes the predictions.

    Creates one folder for each language and one for the global predictions. In each folder creates an overview file,
        one folder for the grouping by predictions and one for the grouping by ground truth. In those folders, we
        create a file for each USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        classification_metrics (AllClassificationMetrics): The classification metrics computed for the predictions.
        out_dir (Path): Path to the output directory.

    """
    out_dir = out_dir / "predictions_per_class"
    # delete and recreate to remove any old files
    if out_dir.exists() and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            key_class_config=KeyClassConfig(
                get_first_key_class=lambda layer: layer.ground_truth_uscs_class,
                get_second_key_class=lambda layer: layer.prediction_uscs_class,
                first_key_str="ground_truth",
                second_key_str="prediction",
                metric_used="recall",
            ),
            metrics_dict=metrics_dict_in_language,
            out_dir=out_dir / language / "group_by_ground_truth",
        )

        # groupby prediction first
        write_per_class_predictions(
            layers_in_language,
            key_class_config=KeyClassConfig(
                get_first_key_class=lambda layer: layer.prediction_uscs_class,
                get_second_key_class=lambda layer: layer.ground_truth_uscs_class,
                first_key_str="prediction",
                second_key_str="ground_truth",
                metric_used="precision",
            ),
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

        rows = []

        for class_ in USCSClasses:
            # filter metrics dict (e.g. extract "CL_ML" for "global_CL_ML_recall")
            class_metrics_dict = {k: v for k, v in metrics_dict.items() if class_.name == "_".join(k.split("_")[1:-1])}

            num_ground_truth = sum(layer.ground_truth_uscs_class == class_ for layer in layers_with_predictions)
            num_pred = sum(layer.prediction_uscs_class == class_ for layer in layers_with_predictions)

            rows.append(
                {
                    "class": class_.name,
                    "f1": next(v for k, v in class_metrics_dict.items() if k.endswith("f1")),
                    "precision": next(v for k, v in class_metrics_dict.items() if k.endswith("precision")),
                    "recall": next(v for k, v in class_metrics_dict.items() if k.endswith("recall")),
                    "number_ground_truth": num_ground_truth,
                    "number_prediction": num_pred,
                }
            )

        # Sort rows by number_ground_truth in descending order
        rows.sort(key=lambda x: x["number_ground_truth"], reverse=True)

        for row in rows:
            writer.writerow(row)


def build_class_stats(
    samples_for_class: list[LayerInformations], first_key_class: USCSClasses, key_class_config: KeyClassConfig
) -> tuple[OrderedDict, dict, int]:
    """Builds detailed statistics and sample data for a specific class.

    Args:
        samples_for_class: Layers matching the first key class.
        first_key_class: The class being analyzed.
        key_class_config: Functions and labels according to the grouping that was used.

    Returns:
        tuple: A tuple containing:
            - Ordered stats dictionary per second class.
            - All samples grouped by second class.
            - Total number of samples for the first class.
    """
    total = len(samples_for_class)
    class_stats = {}
    samples_grouped = {}

    for second_key_class in USCSClasses:
        matched_samples = [
            layer for layer in samples_for_class if key_class_config.get_second_key_class(layer) == second_key_class
        ]

        stat = {
            f"is_{key_class_config.first_key_str}_class": first_key_class == second_key_class,
            f"number_{key_class_config.second_key_str}": len(matched_samples),
            f"proportion_{key_class_config.second_key_str}": len(matched_samples) / total if total else 0.0,
        }

        class_stats[second_key_class.name] = stat

        samples_grouped[second_key_class.name] = [
            {
                "file_name": layer.filename,
                "borehole_index": layer.borehole_index,
                "layer_index": layer.layer_index,
                "language": layer.language,
                "material_description": layer.material_description,
                "ground_truth_uscs_class": layer.ground_truth_uscs_class.name,
                "prediction_uscs_class": layer.prediction_uscs_class.name,
            }
            for layer in matched_samples
        ]

    sorted_stats = OrderedDict(
        sorted(
            class_stats.items(),
            key=lambda x: (
                x[1][f"is_{key_class_config.first_key_str}_class"],
                x[1][f"proportion_{key_class_config.second_key_str}"],
            ),
            reverse=True,
        )
    )

    return sorted_stats, samples_grouped, total


def write_per_class_predictions(
    layers_with_predictions: list[LayerInformations],
    key_class_config: KeyClassConfig,
    metrics_dict: dict[str, float],
    out_dir: Path,
):
    """Creates one file per USCS class and write the predictions and ground truth data for each class.

    Args:
        layers_with_predictions (list[LayerInformations]): List of layers with predictions.
        key_class_config: Functions and labels according to the grouping that should be applied.
        metrics_dict (dict[str, float]): the dict containing the relevent metrics.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    for first_key_class in USCSClasses:
        samples_for_first_key = [
            layer
            for layer in layers_with_predictions
            if key_class_config.get_first_key_class(layer) == first_key_class
        ]

        # get the statistics for each second class, relative to the first_key_class
        stats_first_key, samples_for_first_key, total_first_key = build_class_stats(
            samples_for_first_key, first_key_class, key_class_config
        )

        # get the metric for the correct first key (recall if "zoom" is on ground truth, precision otherwise)
        first_key_metric_micro = next(
            (
                v
                for k, v in metrics_dict.items()
                if key_class_config.metric_used in k and first_key_class.name == "_".join(k.split("_")[1:-1])
            )
        )

        # build final dict
        final_dict = OrderedDict()
        final_dict[f"number_{key_class_config.first_key_str}"] = total_first_key
        final_dict[f"micro_{key_class_config.metric_used}_for_{key_class_config.first_key_str}_class"] = (
            first_key_metric_micro
        )
        final_dict.update(stats_first_key)
        final_dict["samples"] = samples_for_first_key

        output_file = out_dir / f"{key_class_config.first_key_str}_class_{first_key_class.name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)
