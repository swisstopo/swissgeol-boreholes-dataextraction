"""Script to convert ground truth file into the new format."""

import json
from pathlib import Path

import click


@click.command()
@click.option("-g", "--ground_truth_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out_path", type=click.Path(path_type=Path))
def convert_ground_truth(ground_truth_path: Path, out_path: Path) -> dict:
    """Convert the old ground truth format to the new format.

    Note: drops all information that is not the material description. The depth interval is set to None.
    The reason for this is, that old ground truth does not link depth intervals with the material description.

    Args:
        ground_truth_path (Path): Path to the old ground truth file.
        out_path (Path): Path to store the new ground truth file.
    """
    with open(ground_truth_path) as f:
        ground_truth_old = json.load(f)

    ground_truth_new = {}
    for pdf in ground_truth_old:
        layers = []
        for element in ground_truth_old[pdf]:
            if element["tag"] == "Material description":
                layers.append(
                    {"material_description": element["text"], "depth_interval": {"start": None, "end": None}}
                )
        ground_truth_new[pdf] = {"layers": layers}

    with open(out_path, "w") as f:
        json.dump(ground_truth_new, f, indent=4)


if __name__ == "__main__":
    convert_ground_truth()
