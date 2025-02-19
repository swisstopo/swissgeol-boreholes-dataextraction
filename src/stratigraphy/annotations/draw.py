"""This module contains functionalities to draw on pdf pages."""

import logging
import os
from pathlib import Path

import fitz
import pandas as pd
from dotenv import load_dotenv
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBoxes
from stratigraphy.groundwater.groundwater_extraction import Groundwater
from stratigraphy.layer.layer import Layer
from stratigraphy.metadata.coordinate_extraction import Coordinate
from stratigraphy.metadata.elevation_extraction import Elevation
from stratigraphy.util.predictions import OverallFilePredictions

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

colors = ["purple", "blue"]

logger = logging.getLogger(__name__)


def draw_predictions(
    predictions: OverallFilePredictions,
    directory: Path,
    out_directory: Path,
    document_level_metadata_metrics: None | pd.DataFrame,
) -> None:
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
        document_level_metadata_metrics (None | pd.DataFrame): Document level metadata metrics.
    """
    if directory.is_file():  # deal with the case when we pass a file instead of a directory
        directory = directory.parent
    for file_prediction in predictions.file_predictions_list:
        logger.info("Drawing predictions for file %s", file_prediction.file_name)

        bounding_boxes = file_prediction.bounding_boxes
        coordinates = file_prediction.metadata.coordinates
        elevation = file_prediction.metadata.elevation

        # Assess the correctness of the metadata
        if (
            document_level_metadata_metrics is not None
            and file_prediction.file_name in document_level_metadata_metrics.index
        ):
            is_coordinates_correct = document_level_metadata_metrics.loc[file_prediction.file_name].coordinate
            is_elevation_correct = document_level_metadata_metrics.loc[file_prediction.file_name].elevation
        else:
            logger.warning(
                "Metrics for file %s not found in document_level_metadata_metrics.", file_prediction.file_name
            )
            is_coordinates_correct = None
            is_elevation_correct = None

        try:
            with fitz.Document(directory / file_prediction.file_name) as doc:
                for page_index, page in enumerate(doc):
                    page_number = page_index + 1
                    shape = page.new_shape()  # Create a shape object for drawing
                    if page_number == 1:
                        draw_metadata(
                            shape,
                            page.derotation_matrix,
                            page.rotation,
                            coordinates,
                            is_coordinates_correct,
                            elevation,
                            is_elevation_correct,
                        )
                    if coordinates is not None and page_number == coordinates.page:
                        draw_coordinates(shape, coordinates)
                    if elevation is not None and page_number == elevation.page:
                        draw_elevation(shape, elevation)
                    for groundwater_entry in file_prediction.groundwater.groundwater:
                        if page_number == groundwater_entry.page:
                            draw_groundwater(shape, groundwater_entry)
                    draw_depth_columns_and_material_rect(
                        shape,
                        page.derotation_matrix,
                        [bboxes for bboxes in bounding_boxes if bboxes.page == page_number],
                    )
                    draw_material_descriptions(
                        shape,
                        page.derotation_matrix,
                        [
                            layer
                            for layer in file_prediction.layers_in_document.layers
                            if layer.material_description.page == page_number
                        ],
                    )
                    shape.commit()  # Commit all the drawing operations to the page

                    tmp_file_path = out_directory / f"{file_prediction.file_name}_page{page_number}.png"
                    fitz.utils.get_pixmap(page, matrix=fitz.Matrix(2, 2), clip=page.rect).save(tmp_file_path)

                    if mlflow_tracking:  # This is only executed if MLFlow tracking is enabled
                        try:
                            import mlflow

                            mlflow.log_artifact(tmp_file_path, artifact_path="pages")
                        except NameError:
                            logger.warning("MLFlow could not be imported. Skipping logging of artifact.")

        except (FileNotFoundError, fitz.FileDataError) as e:
            logger.error("Error opening file %s: %s", file_prediction.file_name, e)
            continue

        logger.info("Finished drawing predictions for file %s", file_prediction.file_name)


def draw_metadata(
    shape: fitz.Shape,
    derotation_matrix: fitz.Matrix,
    rotation: float,
    coordinates: Coordinate | None,
    is_coordinate_correct: bool | None,
    elevation_info: Elevation | None,
    is_elevation_correct: bool | None,
) -> None:
    """Draw the extracted metadata on the top of the given PDF page.

    The data is always drawn at the top-left of the page, without considering where on the page the data was originally
    found / extracted from.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        rotation (float): The rotation of the page.
        coordinates (Coordinate | None): The coordinate object to draw.
        is_coordinate_correct (bool  | None): Whether the coordinate information is correct.
        elevation_info (Elevation | None): The elevation information to draw.
        is_elevation_correct (bool | None): Whether the elevation information is correct.
    """
    coordinate_rect = fitz.Rect([5, 5, 250, 30])
    elevation_rect = fitz.Rect([5, 30, 250, 55])

    shape.draw_rect(coordinate_rect * derotation_matrix)
    shape.finish(fill=fitz.utils.getColor("gray"), fill_opacity=0.5)
    shape.insert_textbox(coordinate_rect * derotation_matrix, f"Coordinates: {coordinates}", rotate=rotation)
    if is_coordinate_correct is not None:
        # TODO associate correctness with the extracted coordinates in a better way
        coordinate_color = "green" if is_coordinate_correct else "red"
        shape.draw_line(
            coordinate_rect.top_left * derotation_matrix,
            coordinate_rect.bottom_left * derotation_matrix,
        )
        shape.finish(
            color=fitz.utils.getColor(coordinate_color),
            width=6,
            stroke_opacity=0.5,
        )

    # Draw the bounding box around the elevation information
    elevation_txt = f"Elevation: {elevation_info.elevation} m" if elevation_info is not None else "Elevation: N/A"
    shape.draw_rect(elevation_rect * derotation_matrix)
    shape.finish(fill=fitz.utils.getColor("gray"), fill_opacity=0.5)
    shape.insert_textbox(elevation_rect * derotation_matrix, elevation_txt, rotate=rotation)
    if is_elevation_correct is not None:
        elevation_color = "green" if is_elevation_correct else "red"
        shape.draw_line(
            elevation_rect.top_left * derotation_matrix,
            elevation_rect.bottom_left * derotation_matrix,
        )
        shape.finish(
            color=fitz.utils.getColor(elevation_color),
            width=6,
            stroke_opacity=0.5,
        )


def draw_coordinates(shape: fitz.Shape, coordinates: Coordinate) -> None:
    """Draw a bounding box around the area of the page where the coordinates were extracted from.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        coordinates (Coordinate): The coordinate object to draw.
    """
    shape.draw_rect(coordinates.rect)
    shape.finish(color=fitz.utils.getColor("purple"))


def draw_groundwater(shape: fitz.Shape, groundwater_entry: FeatureOnPage[Groundwater]) -> None:
    """Draw a bounding box around the area of the page where the groundwater information was extracted from.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        groundwater_entry (FeatureOnPage[Groundwater]): The groundwater information to draw.
    """
    shape.draw_rect(groundwater_entry.rect)
    shape.finish(color=fitz.utils.getColor("pink"))


def draw_elevation(shape: fitz.Shape, elevation: Elevation) -> None:
    """Draw a bounding box around the area of the page where the elevation were extracted from.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        elevation (Elevation): The elevation information to draw.
    """
    shape.draw_rect(elevation.rect)
    shape.finish(color=fitz.utils.getColor("blue"))


def draw_material_descriptions(shape: fitz.Shape, derotation_matrix: fitz.Matrix, layers: list[Layer]) -> None:
    """Draw information about material descriptions on a pdf page.

    In particular, this function:
        - draws rectangles around the material description text blocks,
        - draws lines connecting the material description text blocks to the depth intervals,
        - colors the lines of the material description text blocks based on whether they were correctly identified.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        layers (LayerPrediction): The predictions for the page.
    """
    for index, layer in enumerate(layers):
        if layer.material_description.rect is not None:
            shape.draw_rect(
                fitz.Rect(layer.material_description.rect) * derotation_matrix,
            )
            shape.finish(color=fitz.utils.getColor("orange"))
        draw_layer(shape=shape, derotation_matrix=derotation_matrix, layer=layer, index=index)


def draw_depth_columns_and_material_rect(
    shape: fitz.Shape, derotation_matrix: fitz.Matrix, bounding_boxes: list[BoundingBoxes]
):
    """Draw depth columns as well as the material rects on a pdf page.

    In particular, this function:
        - draws rectangles around the depth columns,
        - draws rectangles around the depth column entries,
        - draws rectangles around the material description columns.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        bounding_boxes (list[BoundingBoxes]): List of bounding boxes for depth column and material descriptions.
    """
    for bboxes in bounding_boxes:
        if bboxes.sidebar_bbox:  # Draw rectangle for depth columns
            shape.draw_rect(
                fitz.Rect(bboxes.sidebar_bbox.rect) * derotation_matrix,
            )
            shape.finish(color=fitz.utils.getColor("green"))
            for depth_column_entry in bboxes.depth_column_entry_bboxes:  # Draw rectangle for depth column entries
                shape.draw_rect(
                    fitz.Rect(depth_column_entry.rect) * derotation_matrix,
                )
            shape.finish(color=fitz.utils.getColor("purple"))

        shape.draw_rect(  # Draw rectangle for material description column
            bboxes.material_description_bbox.rect * derotation_matrix,
        )
        shape.finish(color=fitz.utils.getColor("red"))


def draw_layer(shape: fitz.Shape, derotation_matrix: fitz.Matrix, layer: Layer, index: int):
    """Draw layers on a pdf page.

    In particular, this function:
        - draws lines connecting the material description text layers to the depth intervals,
        - colors the lines of the material description text layers based on whether they were correctly identified.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        layer (Layer): The layer (depth interval and material description).
        index (int): Index of the layer.
    """
    material_description = layer.material_description.feature
    if material_description.lines:
        color = colors[index % len(colors)]

        # background color for material description
        for line in [line for line in material_description.lines]:
            shape.draw_rect(line.rect * derotation_matrix)
            shape.finish(
                color=fitz.utils.getColor(color),
                fill_opacity=0.2,
                fill=fitz.utils.getColor(color),
                width=0,
            )
            if material_description.is_correct is not None:
                correct_color = "green" if material_description.is_correct else "red"
                shape.draw_line(
                    line.rect.top_left * derotation_matrix,
                    line.rect.bottom_left * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )

        if layer.depths:
            # background color for depth interval
            background_rect = layer.depths.background_rect
            if background_rect is not None:
                shape.draw_rect(
                    background_rect * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(color),
                    fill_opacity=0.2,
                    fill=fitz.utils.getColor(color),
                    width=0,
                )

                # draw green line if depth interval is correct else red line
                if layer.is_correct is not None:
                    depth_is_correct_color = "green" if layer.is_correct else "red"
                    shape.draw_line(
                        background_rect.top_left * derotation_matrix,
                        background_rect.bottom_left * derotation_matrix,
                    )
                    shape.finish(
                        color=fitz.utils.getColor(depth_is_correct_color),
                        width=6,
                        stroke_opacity=0.5,
                    )

            # line from depth interval to material description
            depths_rect = fitz.Rect()
            if layer.depths.start:
                depths_rect.include_rect(layer.depths.start.rect)
            if layer.depths.end:
                depths_rect.include_rect(layer.depths.end.rect)

            if not layer.material_description.rect.contains(depths_rect):
                # Depths are separate from the material description: draw a line connecting them
                line_anchor = layer.depths.line_anchor
                if line_anchor:
                    rect = layer.material_description.rect
                    shape.draw_line(
                        line_anchor * derotation_matrix,
                        fitz.Point(rect.x0, (rect.y0 + rect.y1) / 2) * derotation_matrix,
                    )
                    shape.finish(
                        color=fitz.utils.getColor(color),
                    )
            else:
                # Depths are part of the material description: only draw bounding boxes
                if layer.depths.start:
                    shape.draw_rect(fitz.Rect(layer.depths.start.rect) * derotation_matrix)
                if layer.depths.end:
                    shape.draw_rect(fitz.Rect(layer.depths.end.rect) * derotation_matrix)
                shape.finish(color=fitz.utils.getColor("purple"))
