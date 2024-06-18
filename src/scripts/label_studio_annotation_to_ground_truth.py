"""Script to convert annotations from label studio to a ground truth file."""

import json
from pathlib import Path

import click
from stratigraphy.util.predictions import FilePredictions


@click.command()
@click.option("-a", "--annotation-file-path", type=click.Path(path_type=Path), help="The path to the annotation file.")
@click.option("-o", "--output-path", type=click.Path(path_type=Path), help="The output path of the ground truth file.")
def convert_annotations_to_ground_truth(annotation_file_path: Path, output_path: Path):
    """Convert the annotation file to the ground truth format.

    Args:
        annotation_file_path (Path): The path to the annotation file.
        output_path (Path): The output path of the ground truth file.
    """
    with open(annotation_file_path) as f:
        annotations = json.load(f)

    file_predictions = FilePredictions.create_from_label_studio(annotations)

    ground_truth = {}
    for prediction in file_predictions:
        ground_truth = {**ground_truth, **prediction.convert_to_ground_truth()}

    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=4)


if __name__ == "__main__":
    convert_annotations_to_ground_truth()
