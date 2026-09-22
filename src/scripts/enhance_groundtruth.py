"""Script to enhance a ground truth file with predicted labels from classification JSON outputs.

Usage:
python src/scripts/enhance_groundtruth.py \
  -g Path to ground truth file which should be enhanced  \
  -p class prediction json  \
  -p optional further class prediction json
  -o optional path to output file (default: <ground_truth_stem>_enhanced.json next to the input file)
"""

import json
import logging
from pathlib import Path
from typing import get_args, get_origin

import click

from classification.utils.datasets import ExistingClassificationSystems
from core.ground_truth import GroundTruth, GroundTruthConsolidated, GroundTruthUnconsolidated

logger = logging.getLogger(__name__)

_SUB_MODEL_CLASSES: dict[str, type] = {
    "consolidated": GroundTruthConsolidated,
    "unconsolidated": GroundTruthUnconsolidated,
}


def _is_list_field(sub_model_cls: type, field_name: str) -> bool:
    """Return True if the Pydantic field is annotated as a list type.

    Args:
        sub_model_cls: The Pydantic model class containing the field.
        field_name: The field name to inspect.

    Returns:
        bool: True when the field holds a list, False for scalar fields.
    """
    annotation = sub_model_cls.model_fields[field_name].annotation
    return any(get_origin(arg) is list for arg in get_args(annotation))


def _apply_predictions(ground_truth: GroundTruth, predictions_path: Path) -> int:
    """Write predicted labels from a class predictions JSON into the ground truth in-place.

    The target sub-model (consolidated / unconsolidated) and field are resolved from
    the classification system declared inside the predictions file.

    Args:
        ground_truth: GroundTruth object to update.
        predictions_path: Path to a class predictions JSON file.

    Returns:
        int: Number of layers updated.
    """
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    if not predictions:
        return 0

    class_system_name = predictions[0]["class_system"]
    classification_system = ExistingClassificationSystems.get_classification_system_type(class_system_name)
    sub_model_key, field_name = classification_system.get_layer_ground_truth_keys()

    sub_model_cls = _SUB_MODEL_CLASSES[sub_model_key]
    is_list = _is_list_field(sub_model_cls, field_name)

    updated = 0
    for entry in predictions:
        prediction = entry.get("prediction_class")
        if not prediction:
            continue

        boreholes = ground_truth.for_file(entry["filename"]) or []
        borehole = next((b for b in boreholes if b.borehole_index == entry["borehole_index"]), None)
        if borehole is None or entry["layer_index"] >= len(borehole.layers):
            continue

        layer = borehole.layers[entry["layer_index"]]
        values = prediction if isinstance(prediction, list) else [prediction]
        value = values if is_list else values[0]

        sub_model = getattr(layer, sub_model_key) or sub_model_cls()
        setattr(layer, sub_model_key, sub_model.model_copy(update={field_name: value}))
        updated += 1

    logger.info("Applied %d predictions from %s (%s).", updated, predictions_path.name, class_system_name)
    return updated


@click.command(help="Enhance a ground truth file with predicted labels from one or more prediction JSON files.")
@click.option(
    "-g",
    "--ground-truth",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the input ground truth JSON file.",
)
@click.option(
    "-p",
    "--predictions",
    "predictions_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Path to a class predictions JSON file. Repeat to apply multiple files.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path. Defaults to <ground_truth_stem>_enhanced.json next to the input file.",
)
def enhance(ground_truth: Path, predictions_paths: tuple[Path, ...], output: Path | None) -> None:
    """Enhance a ground truth file with predicted labels from classification outputs.

    Args:
        ground_truth (Path): Path to the input ground truth JSON file.
        predictions_paths (tuple[Path, ...]): Paths to class predictions JSON files.
        output (Path | None): Output path for the enhanced ground truth.
    """
    output = output or ground_truth.parent / f"{ground_truth.stem}_enhanced{ground_truth.suffix}"

    gt = GroundTruth(ground_truth)
    for predictions_path in predictions_paths:
        _apply_predictions(gt, predictions_path)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(gt.to_json(), f, ensure_ascii=False, indent=4)

    logger.info("Enhanced ground truth saved to %s.", output)


if __name__ == "__main__":
    enhance()
