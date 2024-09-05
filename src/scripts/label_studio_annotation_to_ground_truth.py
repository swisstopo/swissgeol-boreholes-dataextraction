"""Script to convert annotations from label studio to a ground truth file."""

import contextlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import fitz
from stratigraphy.coordinates.coordinate_extraction import Coordinate
from stratigraphy.util.interval import AnnotatedInterval
from stratigraphy.util.predictions import BoreholeMetaData, FilePredictions, LayerPrediction
from stratigraphy.util.textblock import MaterialDescription

logger = logging.getLogger(__name__)


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

    file_predictions = create_from_label_studio(annotations)

    ground_truth = {}
    for prediction in file_predictions:
        ground_truth = {**ground_truth, **prediction.convert_to_ground_truth()}

    # check if the output path exists
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=4)


def create_from_label_studio(annotation_results: dict) -> list[FilePredictions]:
    """Create predictions class for a file given the annotation results from Label Studio.

    This method is meant to import annotations from label studio. The primary use case is to
    use the annotated data for evaluation. For that purpose, there is the convert_to_ground_truth
    method, which then converts the predictions to ground truth format.

    NOTE: We may want to adjust this method to return a single instance of the class,
    instead of a list of class objects.

    Args:
        annotation_results (dict): The annotation results from Label Studio.
                                   The annotation_results can cover multiple files.

    Returns:
        list[FilePredictions]: A list of FilePredictions objects, one for each file present in the
                               annotation_results.
    """
    file_predictions = defaultdict(list)
    metadata = {}
    for annotation in annotation_results:
        # get page level information
        file_name, _ = _get_file_name_and_page_index(annotation)
        page_width = annotation["annotations"][0]["result"][0]["original_width"]
        page_height = annotation["annotations"][0]["result"][0]["original_height"]

        # extract all material descriptions and depth intervals and link them together
        # Note: we need to loop through the annotations twice, because the order of the annotations is
        # not guaranteed. In the first iteration we grasp all IDs, in the second iteration we extract the
        # information for each id.
        material_descriptions = {}
        depth_intervals = {}
        coordinates = {}
        linking_objects = []

        # define all the material descriptions and depth intervals with their ids
        for annotation_result in annotation["annotations"][0]["result"]:
            if annotation_result["type"] == "labels":
                if annotation_result["value"]["labels"] == ["Material Description"]:
                    material_descriptions[annotation_result["id"]] = {
                        "rect": annotation_result["value"]
                    }  # TODO extract rectangle properly; does not impact the ground truth though.
                elif annotation_result["value"]["labels"] == ["Depth Interval"]:
                    depth_intervals[annotation_result["id"]] = {}
                elif annotation_result["value"]["labels"] == ["Coordinates"]:
                    coordinates[annotation_result["id"]] = {}
            if annotation_result["type"] == "relation":
                linking_objects.append({"from_id": annotation_result["from_id"], "to_id": annotation_result["to_id"]})

        # check annotation results for material description or depth interval ids
        for annotation_result in annotation["annotations"][0]["result"]:
            with contextlib.suppress(KeyError):
                id = annotation_result["id"]  # relation regions do not have an ID.
            if annotation_result["type"] == "textarea":
                if id in material_descriptions:
                    material_descriptions[id]["text"] = annotation_result["value"]["text"][
                        0
                    ]  # There is always only one element. TO CHECK!
                    if len(annotation_result["value"]["text"]) > 1:
                        print(f"More than one text in material description: {annotation_result['value']['text']}")
                elif id in depth_intervals:
                    depth_interval_text = annotation_result["value"]["text"][0]
                    start, end = _get_start_end_from_text(depth_interval_text)
                    depth_intervals[id]["start"] = start
                    depth_intervals[id]["end"] = end
                    depth_intervals[id]["background_rect"] = annotation_result[
                        "value"
                    ]  # TODO extract rectangle properly; does not impact the ground truth though.
                elif id in coordinates:
                    coordinates[id]["text"] = annotation_result["value"]["text"][0]
                else:
                    print(f"Unknown id: {id}")

        # create the layer prediction objects by linking material descriptions with depth intervals
        layers = []

        for link in linking_objects:
            from_id = link["from_id"]
            to_id = link["to_id"]
            material_description_prediction = MaterialDescription(**material_descriptions.pop(from_id))
            depth_interval_prediction = AnnotatedInterval(**depth_intervals.pop(to_id))
            layers.append(
                LayerPrediction(
                    material_description=material_description_prediction,
                    depth_interval=depth_interval_prediction,
                    material_is_correct=True,
                    depth_interval_is_correct=True,
                )
            )

        if material_descriptions or depth_intervals:
            # TODO: This should not be acceptable. Raising an error doesnt seem the right way to go either.
            # But at least it should be warned.
            print("There are material descriptions or depth intervals left over.")
            print(material_descriptions)
            print(depth_intervals)

        # instantiate metadata object
        if coordinates:
            coordinate_text = coordinates.popitem()[1]["text"]
            # TODO: we could extract the rectangle as well. For conversion to ground truth this does not matter.
            metadata[file_name] = BoreholeMetaData(coordinates=_get_coordinates_from_text(coordinate_text))

        # create the page prediction object
        if file_name in file_predictions:
            # append the page predictions to the existing file predictions
            file_predictions[file_name].layers.extend(layers)
            file_predictions[file_name].page_sizes.append({"width": page_width, "height": page_height})
        else:
            # create a new file prediction object if it does not exist yet
            file_predictions[file_name] = FilePredictions(
                layers=layers,
                file_name=f"{file_name}.pdf",
                language="unknown",
                metadata=metadata.get(file_name),
                groundwater_entries=[],
                depths_materials_columns_pairs=[],
                page_sizes=[{"width": page_width, "height": page_height}],
            )

    file_predictions_list = []
    for _, file_prediction in file_predictions.items():
        file_predictions_list.append(file_prediction)  # TODO: language should not be required here.

    return file_predictions_list


def _get_coordinates_from_text(text: str) -> Coordinate | None:
    """Convert a string to a Coordinate object.

    The string has the format: E: 498'561, N: 114'332 or E: 2'498'561, N: 1'114'332.

    Args:
        text (str): The input string to be converted to a Coordinate object.

    Returns:
        Coordinate: The Coordinate object.
    """
    try:
        east_text, north_text = text.split(", ")
        east = int(east_text.split(": ")[1].replace("'", ""))
        north = int(north_text.split(": ")[1].replace("'", ""))
        return Coordinate.from_values(east=east, north=north, page=0, rect=fitz.Rect([0, 0, 0, 0]))
    except ValueError:  # This is likely due to a wrong format of the text.
        logger.warning(f"Could not extract coordinates from text: {text}.")
        return None


def _get_start_end_from_text(text: str) -> tuple[float, float]:
    start, end = text.split("end: ")
    start = start.split("start: ")[1]
    return float(start), float(end)


def _get_file_name_and_page_index(annotation: dict[str, Any]) -> tuple[str, int]:
    """Extract the file name and page index from the annotation.

    Args:
        annotation (dict): The annotation dictionary. Exported from Label Studio.

    Returns:
        tuple[str, int]: The file name and the page index (zero-based).
    """
    file_name = annotation["data"]["ocr"].split("/")[-1]
    file_name = file_name.split(".")[0]
    return file_name.split("_")


if __name__ == "__main__":
    convert_annotations_to_ground_truth()
