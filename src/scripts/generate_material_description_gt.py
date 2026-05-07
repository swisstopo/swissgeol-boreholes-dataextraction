"""Split a directory of PDFs into train/validation/test using a deterministic hash."""

import hashlib
import json
import logging
from pathlib import Path

import click

from extraction.features.predictions.overall_file_predictions import OverallFilePredictions

logger = logging.getLogger(__name__)


def deterministic_hash_ratio(text: str) -> float:
    """Map a string deterministically to a float in [0, 1).

    This is used to assign files to splits in a reproducible way, based only on
    their filename (or any stable string key).

    Args:
        text: Input string to hash (e.g., a filename).

    Returns:
        A float in the half-open interval [0, 1).
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Use the first 8 bytes (64 bits) to build a stable ratio in [0, 1).
    return int.from_bytes(h[:8], "big") / 2**64


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
    with open(prediction, encoding="utf8") as f:
        predictions = OverallFilePredictions.from_json(json.load(f))
    # gt = GroundTruth(ground_truth)

    borehole = predictions.file_predictions_list[0].boreholes[0]
    layer = borehole.layers_in_borehole.layers[0]

    print(borehole.metadata.name.feature.is_correct)
    print(borehole.metadata.coordinates.feature.is_correct)
    print(borehole.metadata.elevation.feature.is_correct)

    print(layer.material_description.is_correct)
    print(layer.depths.is_correct)
    print(layer.is_correct)

    logger.info("Done.")


if __name__ == "__main__":
    generate()
