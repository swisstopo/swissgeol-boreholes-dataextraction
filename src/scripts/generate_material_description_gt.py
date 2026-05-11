"""TODO."""

import json
import logging
from pathlib import Path

import click

from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions

logger = logging.getLogger(__name__)


@click.command(help="Generate material description from prediction")
@click.option(
    "-p",
    "--prediction",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prediction file.",
)
@click.option(
    "-g",
    "--ground-truth",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ground truth to update.",
)
def generate(prediction: Path, ground_truth: Path) -> None:
    """TOOD."""
    new_ground_truth = ground_truth.parent / f"{ground_truth.stem}.new{ground_truth.suffix}"
    with open(prediction, encoding="utf8") as f:
        predictions = OverallFilePredictions.from_json(json.load(f))
    gts = GroundTruth(ground_truth)

    for file_prediction in predictions.file_predictions_list:
        gt_for_prediction = gts.for_file(file_prediction.filename)
        if not gt_for_prediction:
            continue
        # Check layers that have correct prediction
        correct_layers = [
            layer
            for borehole in file_prediction.boreholes
            for layer in borehole.layers_in_borehole.layers
            if layer.depths.is_correct
        ]

        # Match correct layers to GT (assume single borehole)
        for layer_gt in gt_for_prediction[0].layers:
            if matched_layer := list(
                filter(lambda x: layer_gt.depth_interval.start == x.depths.start.value, correct_layers)
            ):
                layer_gt.material_description = matched_layer[0].material_description.text

    # Count the number of material descriptions
    with open(new_ground_truth, "w", encoding="utf8") as f:
        json.dump(gts.to_json(), f, indent=4)

    logger.info("Done.")


if __name__ == "__main__":
    generate()
