"""This module contains functionalities to draw on pdf pages."""

import logging
import os
import time
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stratigraphy.util.coordinate_extraction import Coordinate
from stratigraphy.util.interval import BoundaryInterval
from stratigraphy.util.predictions import FilePredictions, LayerPrediction
from stratigraphy.util.textblock import TextBlock

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

colors = ["purple", "blue"]

logger = logging.getLogger(__name__)


def draw_predictions(predictions: list[FilePredictions], directory: Path, out_directory: Path) -> pd.DataFrame:
    """Draw predictions on pdf pages.

    Draws various recognized information on the pdf pages present at directory and saves
    them as images in the out_directory.

    The drawn information includes:
        - Depth columns (if available)
        - Depth column entries (if available)
        - Material description columns
        - Material description text blocks
        - Color-coded correctness of the material description text blocks
        - Assignments of material description text blocks to depth intervals (if available)

    Args:
        predictions (dict): Content of the predictions.json file.
        directory (Path): Path to the directory containing the pdf files.
        out_directory (Path): Path to the output directory where the images are saved.
    """
    drawing_times = {
        "filename": [],
        "draw_time": [],
        "layers": [],
        "lines": [],
    }

    if directory.is_file():  # deal with the case when we pass a file instead of a directory
        directory = directory.parent
    for file_name, file_prediction in predictions.items():
        start_time = time.time()

        logger.info("Drawing predictions for file %s", file_name)
        with fitz.Document(directory / file_name) as doc:
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                depths_materials_column_pairs = file_prediction.depths_materials_columns_pairs
                if page_index == 0:
                    draw_metadata(
                        page,
                        file_prediction.metadata.coordinates,
                        file_prediction.metadata_is_correct.get("coordinates"),
                    )
                    print("time to draw metadata", time.time() - start_time)
                if file_prediction.metadata.coordinates is not None:
                    draw_coordinates(page, file_prediction.metadata.coordinates)
                    print("time to draw coordinates", time.time() - start_time)
                draw_depth_columns_and_material_rect(page, depths_materials_column_pairs)
                print("time to draw depth columns and material rects", time.time() - start_time)
                draw_material_descriptions(page, file_prediction.layers)
                print("time to draw material descriptions", time.time() - start_time)

                tmp_file_path = out_directory / f"{file_name}_page{page_number}.png"
                fitz.utils.get_pixmap(page, matrix=fitz.Matrix(2, 2), clip=page.rect).save(tmp_file_path)

                if mlflow_tracking:  # This is only executed if MLFlow tracking is enabled
                    try:
                        import mlflow

                        mlflow.log_artifact(tmp_file_path, artifact_path="pages")
                    except NameError:
                        logger.warning("MLFlow could not be imported. Skipping logging of artifact.")

        drawing_times["filename"].append(file_name)
        drawing_times["draw_time"].append(time.time() - start_time)
        drawing_times["layers"].append(len(file_prediction.layers))
        drawing_times["lines"].append(
            np.sum([len(layer.material_description.lines) for layer in file_prediction.layers])
        )

        logger.info("Finished drawing predictions for file %s", file_name)
        logger.info("Drawing time: %s", time.time() - start_time)

    return pd.DataFrame(drawing_times)


def draw_metadata(page: fitz.Page, coordinates: Coordinate | None, coordinates_is_correct: bool) -> None:
    """Draw the extracted metadata on the top of the given PDF page.

    The data is always drawn at the top-left of the page, without considering where on the page the data was originally
    found / extracted from.

    Args:
        page (fitz.Page): The page to draw on.
        coordinates (Coordinate | None): The coordinate object to draw.
        coordinates_is_correct (bool): Whether the coordinates are correct.
    """
    color = "green" if coordinates_is_correct else "red"
    coordinate_rect = fitz.Rect([5, 5, 200, 25])

    shape = page.new_shape()  # Create a shape object for drawing

    shape.draw_rect(coordinate_rect * page.derotation_matrix)
    shape.finish(fill=fitz.utils.getColor("gray"), fill_opacity=0.5)
    page.insert_htmlbox(coordinate_rect * page.derotation_matrix, f"Coordinates: {coordinates}", rotate=page.rotation)
    shape.draw_line(
        coordinate_rect.top_left * page.derotation_matrix,
        coordinate_rect.bottom_left * page.derotation_matrix,
    )
    shape.finish(
        color=fitz.utils.getColor(color),
        width=6,
        stroke_opacity=0.5,
    )
    shape.commit()  # Commit all the drawing operations to the page


def draw_coordinates(page: fitz.Page, coordinates: Coordinate) -> None:
    """Draw a bounding box around the area of the page where the coordinates were extracted from.

    Args:
        page (fitz.Page): The page to draw on.
        coordinates (Coordinate): The coordinate object to draw.
    """
    shape = page.new_shape()  # Create a shape object for drawing
    if (page.number + 1) == coordinates.page:  ## page.number is 0-based
        shape.draw_rect(coordinates.rect)
        shape.finish(color=fitz.utils.getColor("purple"))
    shape.commit()  # Commit all the drawing operations to the page


def draw_material_descriptions(page: fitz.Page, layers: LayerPrediction) -> None:
    """Draw information about material descriptions on a pdf page.

    In particular, this function:
        - draws rectangles around the material description text blocks,
        - draws lines connecting the material description text blocks to the depth intervals,
        - colors the lines of the material description text blocks based on whether they were correctly identified.

    Args:
        page (fitz.Page): The page to draw on.
        layers (LayerPrediction): The predictions for the page.
    """
    page_number = page.number + 1
    shape = page.new_shape()  # Create a shape object for drawing

    for index, layer in enumerate(layers):
        if layer.material_description.page_number == page_number:
            if layer.material_description.rect is not None:
                shape.draw_rect(
                    fitz.Rect(layer.material_description.rect) * page.derotation_matrix,
                )
                shape.finish(color=fitz.utils.getColor("orange"))
            draw_layer(
                shape=shape,
                derotation_matrix=page.derotation_matrix,
                interval=layer.depth_interval,  # None if no depth interval
                layer=layer.material_description,
                index=index,
                is_correct=layer.material_is_correct,  # None if no ground truth
                depth_is_correct=layer.depth_interval_is_correct,  # None if no ground truth
            )
    shape.commit()  # Commit all the drawing operations to the page


def draw_depth_columns_and_material_rect(page: fitz.Page, depths_materials_column_pairs: list) -> fitz.Page:
    """Draw depth columns as well as the material rects on a pdf page.

    In particular, this function:
        - draws rectangles around the depth columns,
        - draws rectangles around the depth column entries,
        - draws rectangles around the material description columns.

    Args:
        page (fitz.Page): The page to draw on.
        depths_materials_column_pairs (list): List of depth column entries.

    Returns:
        fitz.Page: The page with the drawn depth columns and material rects.
    """
    shape = page.new_shape()  # Create a shape object for drawing
    derotation_matrix = page.derotation_matrix

    for pair in depths_materials_column_pairs:
        depth_column = pair["depth_column"]
        material_description_rect = pair["material_description_rect"]
        page_number = pair["page"]

        # Skip if the page number does not match to display the correct page
        if page_number != page.number + 1:
            continue

        if depth_column is not None:  # Draw rectangle for depth columns
            shape.draw_rect(
                fitz.Rect(depth_column["rect"]) * derotation_matrix,
            )
            shape.finish(color=fitz.utils.getColor("green"))
            for depth_column_entry in depth_column["entries"]:  # Draw rectangle for depth column entries
                shape.draw_rect(
                    fitz.Rect(depth_column_entry["rect"]) * derotation_matrix,
                )
            shape.finish(color=fitz.utils.getColor("purple"))

        shape.draw_rect(  # Draw rectangle for material description column
            fitz.Rect(material_description_rect) * derotation_matrix,
        )
        shape.finish(color=fitz.utils.getColor("red"))

    shape.commit()  # Commit all the drawing operations to the page

    return page


def draw_layer(
    shape: fitz.Shape,
    derotation_matrix: fitz.Matrix,
    interval: BoundaryInterval | None,
    layer: TextBlock,
    index: int,
    is_correct: bool,
    depth_is_correct: bool,
):
    """Draw layers on a pdf page.

    In particular, this function:
        - draws lines connecting the material description text layers to the depth intervals,
        - colors the lines of the material description text layers based on whether they were correctly identified.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        interval (BoundaryInterval | None): Depth interval for the layer.
        layer (MaterialDescriptionPrediction): Material description block for the layer.
        index (int): Index of the layer.
        is_correct (bool): Whether the text block was correctly identified.
        depth_is_correct (bool): Whether the depth interval was correctly identified.
    """
    if layer.lines:
        layer_rect = fitz.Rect(layer.rect)
        color = colors[index % len(colors)]

        # background color for material description
        for line in [line for line in layer.lines]:
            start_time = time.time()
            shape.draw_rect(line.rect * derotation_matrix)
            shape.finish(
                color=fitz.utils.getColor(color),
                fill_opacity=0.2,
                fill=fitz.utils.getColor(color),
                width=0,
            )
            print("time to draw rect - 1", time.time() - start_time)
            if is_correct is not None:
                correct_color = "green" if is_correct else "red"
                start_time = time.time()
                shape.draw_line(
                    line.rect.top_left * derotation_matrix,
                    line.rect.bottom_left * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )
                print("time to draw line - 1", time.time() - start_time)

        if interval:
            # background color for depth interval
            background_rect = interval.background_rect
            if background_rect is not None:
                start_time = time.time()
                shape.draw_rect(
                    background_rect * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(color),
                    fill_opacity=0.2,
                    fill=fitz.utils.getColor(color),
                    width=0,
                )
                print("time to draw rect - 2", time.time() - start_time)

                # draw green line if depth interval is correct else red line
                if depth_is_correct is not None:
                    depth_is_correct_color = "green" if depth_is_correct else "red"
                    start_time = time.time()
                    shape.draw_line(
                        background_rect.top_left * derotation_matrix,
                        background_rect.bottom_left * derotation_matrix,
                    )
                    shape.finish(
                        color=fitz.utils.getColor(depth_is_correct_color),
                        width=6,
                        stroke_opacity=0.5,
                    )
                    print("time to draw line - 2", time.time() - start_time)

            # line from depth interval to material description
            line_anchor = interval.line_anchor
            if line_anchor:
                start_time = time.time()
                shape.draw_line(
                    line_anchor * derotation_matrix,
                    fitz.Point(layer_rect.x0, (layer_rect.y0 + layer_rect.y1) / 2) * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(color),
                )
                print("time to draw line - 3", time.time() - start_time)

    shape.commit()  # Commit all the drawing operations to the page
