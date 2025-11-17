"""Data utils module."""

import csv
import json
import logging
import os
import shutil
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from classification.evaluation.evaluate import AllClassificationMetrics
from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformation
from classification.utils.file_utils import read_params

config_path = "config"
classification_params = read_params("classification_params.yml", config_path)

logger = logging.getLogger(__name__)


def get_data_language_count(layer_descriptions: list[LayerInformation]) -> dict[str, int]:
    """Returns the count of sample for each language.

    Args:
        layer_descriptions (list[LayerInformation]): All the layers.

    Returns:
        dict[str,int]: the count for each language.
    """
    language_counts = dict(Counter(layer.language for layer in layer_descriptions))
    return language_counts


def get_data_class_count(layer_descriptions: list[LayerInformation]) -> dict[str, int]:
    """Returns the count of sample for each class.

    Args:
        layer_descriptions (list[LayerInformation]): All the layers.

    Returns:
        dict[str,int]: the count for each class.
    """
    class_counts = dict(Counter(layer.ground_truth_class.name for layer in layer_descriptions))
    return class_counts


def write_predictions(
    layers_with_predictions: list[LayerInformation], out_dir: Path, out_path: str = "class_predictions.json"
):
    """Writes the predictions and ground truth data to a JSON file.

    Args:
        layers_with_predictions (list[LayerInformation]): List of layers with predictions.
        out_dir (Path): Path to the output directory.
        out_path (str): Name of the output file (default: "class_predictions.json").
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
                "class_system": layer.class_system.get_layer_ground_truth_key(),
                "ground_truth_class": layer.ground_truth_class.name if layer.ground_truth_class is not None else None,
                "prediction_class": layer.prediction_class.name if layer.prediction_class is not None else None,
                "llm_reasoning": layer.llm_reasoning if layer.llm_reasoning is not None else None,
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
            logger.warning(f"Existing file {failures_path} contained invalid JSON and will be overwritten")

    all_failures = existing_failures + api_failures

    with open(failures_path, "w") as f:
        json.dump(all_failures, f, indent=2)

    logger.warning(
        f"Recorded {len(api_failures)} failed API calls to {failures_path}, total failed records: {len(all_failures)}"
    )


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

    get_first_key_class: Callable[[LayerInformation], ClassificationSystem.EnumMember]
    get_second_key_class: Callable[[LayerInformation], ClassificationSystem.EnumMember]
    first_key_str: str
    second_key_str: str
    metric_used: str


def write_per_language_per_class_predictions(
    layers_with_predictions: list[LayerInformation], classification_metrics: AllClassificationMetrics, out_dir: Path
):
    """Creates json files that summarizes the predictions.

    Creates one folder for each language and one for the global predictions. In each folder creates an overview file,
        one folder for the grouping by predictions and one for the grouping by ground truth. In those folders, we
        create a file for each class seen in the data and write the predictions and ground truth data for each.

    Args:
        layers_with_predictions (list[LayerInformation]): List of layers with predictions.
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
                get_first_key_class=lambda layer: layer.ground_truth_class,
                get_second_key_class=lambda layer: layer.prediction_class,
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
                get_first_key_class=lambda layer: layer.prediction_class,
                get_second_key_class=lambda layer: layer.ground_truth_class,
                first_key_str="prediction",
                second_key_str="ground_truth",
                metric_used="precision",
            ),
            metrics_dict=metrics_dict_in_language,
            out_dir=out_dir / language / "group_by_prediction",
        )


def write_overview(metrics_dict: dict[str, float], layers_with_predictions: list[LayerInformation], out_dir: Path):
    """Write an overview CSV containing performance metrics for each class.

    Args:
        metrics_dict (dict[str, float]): Flat dictionary with metrics, e.g., 'global_CL_f1': 0.5.
        layers_with_predictions (list[LayerInformation]): List of layer predictions.
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

        ground_truths = [layer.ground_truth_class for layer in layers_with_predictions]
        predictions = [layer.prediction_class for layer in layers_with_predictions]

        all_classes = set(predictions) | set(ground_truths)

        for class_ in all_classes:
            # filter metrics dict (e.g. extract "CL_ML" for "global_CL_ML_recall")
            class_metrics_dict = {k: v for k, v in metrics_dict.items() if class_.name == "_".join(k.split("_")[1:-1])}

            num_pred = sum(pred == class_ for pred in predictions)
            num_ground_truth = sum(gt == class_ for gt in ground_truths)

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


def write_per_class_predictions(
    layers_with_predictions: list[LayerInformation],
    key_class_config: KeyClassConfig,
    metrics_dict: dict[str, float],
    out_dir: Path,
):
    """Creates one file for each class seen in the data and write the predictions and ground truth data for each.

    Args:
        layers_with_predictions (list[LayerInformation]): List of layers with predictions.
        key_class_config (KeyClassConfig): Functions and labels according to the grouping that should be applied.
        metrics_dict (dict[str, float]): the dict containing the relevent metrics.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    all_first_key_classes = set([key_class_config.get_first_key_class(layer) for layer in layers_with_predictions])

    for first_key_class in all_first_key_classes:
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
                if k.endswith(key_class_config.metric_used) and first_key_class.name == "_".join(k.split("_")[1:-1])
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


def build_class_stats(
    samples_for_class: list[LayerInformation],
    first_key_class: ClassificationSystem.EnumMember,
    key_class_config: KeyClassConfig,
) -> tuple[OrderedDict, dict, int]:
    """Builds detailed statistics and sample data for a specific class.

    Args:
        samples_for_class (list[LayerInformation]): Layers matching the first key class.
        first_key_class (ClassificationSystem.EnumMember): The class being analyzed.
        key_class_config (KeyClassConfig): Functions and labels according to the grouping that was used.

    Returns:
        tuple: A tuple containing:
            - Ordered stats dictionary per second class.
            - All samples grouped by second class.
            - Total number of samples for the first class.
    """
    total = len(samples_for_class)
    class_stats = {}
    samples_grouped = {}

    all_second_key_classes = set([key_class_config.get_second_key_class(layer) for layer in samples_for_class])

    for second_key_class in all_second_key_classes:
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
                "class_system": layer.class_system.get_layer_ground_truth_key(),
                "material_description": layer.material_description,
                "ground_truth_class": layer.ground_truth_class.name,
                "prediction_class": layer.prediction_class.name,
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
