"""This module contains functionalities to draw on pdf pages."""

import logging
import os
from pathlib import Path

import fitz
import pandas as pd
from dotenv import load_dotenv
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depthcolumn.depthcolumn import DepthColumn
from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPairs
from stratigraphy.groundwater.groundwater_extraction import Groundwater
from stratigraphy.layer.layer import Layer
from stratigraphy.metadata.coordinate_extraction import Coordinate
from stratigraphy.metadata.elevation_extraction import Elevation
from stratigraphy.text.textblock import TextBlock
from stratigraphy.util.interval import BoundaryInterval
from stratigraphy.util.predictions import OverallFilePredictions

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

colors = ["purple", "blue"]

logger = logging.getLogger(__name__)


def draw_predictions(
    predictions: OverallFilePredictions,
    directory: Path,
    out_directory: Path,
    document_level_metadata_metrics: pd.DataFrame,
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
        document_level_metadata_metrics (pd.DataFrame): Document level metadata metrics.
    """
    if directory.is_file():  # deal with the case when we pass a file instead of a directory
        directory = directory.parent
    for file_prediction in predictions.file_predictions_list:
        logger.info("Drawing predictions for file %s", file_prediction.file_name)

        depths_materials_column_pairs = file_prediction.depths_materials_columns_pairs
        coordinates = file_prediction.metadata.coordinates
        elevation = file_prediction.metadata.elevation

        # Assess the correctness of the metadata
        if file_prediction.file_name in document_level_metadata_metrics.index:
            is_coordinates_correct = document_level_metadata_metrics.loc[file_prediction.file_name].coordinate
            is_elevation_correct = document_level_metadata_metrics.loc[file_prediction.file_name].elevation
        else:
            logger.warning(
                "Metrics for file %s not found in document_level_metadata_metrics.", file_prediction.file_name
            )
            is_coordinates_correct = False
            is_elevation_correct = False

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
                        [pair for pair in depths_materials_column_pairs if pair.page == page_number],
                    )
                    draw_material_descriptions(
                        shape,
                        page.derotation_matrix,
                        [
                            layer
                            for layer in file_prediction.layers.get_all_layers()
                            if layer.material_description.page_number == page_number
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
    is_coordinate_correct: bool,
    elevation_info: Elevation | None,
    is_elevation_correct: bool,
) -> None:
    """Draw the extracted metadata on the top of the given PDF page.

    The data is always drawn at the top-left of the page, without considering where on the page the data was originally
    found / extracted from.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        rotation (float): The rotation of the page.
        coordinates (Coordinate | None): The coordinate object to draw.
        is_coordinate_correct (Metrics): Whether the coordinate information is correct.
        elevation_info (ElevationInformation | None): The elevation information to draw.
        is_elevation_correct (Metrics): Whether the elevation information is correct.
    """
    # TODO associate correctness with the extracted coordinates in a better way
    coordinate_color = "green" if is_coordinate_correct else "red"
    coordinate_rect = fitz.Rect([5, 5, 250, 30])

    elevation_color = "green" if is_elevation_correct else "red"
    elevation_rect = fitz.Rect([5, 30, 250, 55])

    shape.draw_rect(coordinate_rect * derotation_matrix)
    shape.finish(fill=fitz.utils.getColor("gray"), fill_opacity=0.5)
    shape.insert_textbox(coordinate_rect * derotation_matrix, f"Coordinates: {coordinates}", rotate=rotation)
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
        draw_layer(
            shape=shape,
            derotation_matrix=derotation_matrix,
            interval=layer.depth_interval,  # None if no depth interval
            layer=layer.material_description,
            index=index,
            is_correct=layer.material_is_correct,  # None if no ground truth
            depth_is_correct=layer.depth_interval_is_correct,  # None if no ground truth
        )


def draw_depth_columns_and_material_rect(
    shape: fitz.Shape, derotation_matrix: fitz.Matrix, depths_materials_column_pairs: list[DepthsMaterialsColumnPairs]
):
    """Draw depth columns as well as the material rects on a pdf page.

    In particular, this function:
        - draws rectangles around the depth columns,
        - draws rectangles around the depth column entries,
        - draws rectangles around the material description columns.

    Args:
        shape (fitz.Shape): The shape object for drawing.
        derotation_matrix (fitz.Matrix): The derotation matrix of the page.
        depths_materials_column_pairs (list): List of depth column entries.
    """
    for pair in depths_materials_column_pairs:
        depth_column: DepthColumn = pair.depth_column
        material_description_rect = pair.material_description_rect

        if depth_column:  # Draw rectangle for depth columns
            shape.draw_rect(
                fitz.Rect(depth_column.rect()) * derotation_matrix,
            )
            shape.finish(color=fitz.utils.getColor("green"))
            for depth_column_entry in depth_column.entries:  # Draw rectangle for depth column entries
                shape.draw_rect(
                    fitz.Rect(depth_column_entry.rect) * derotation_matrix,
                )
            shape.finish(color=fitz.utils.getColor("purple"))

        shape.draw_rect(  # Draw rectangle for material description column
            fitz.Rect(material_description_rect) * derotation_matrix,
        )
        shape.finish(color=fitz.utils.getColor("red"))


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
            shape.draw_rect(line.rect * derotation_matrix)
            shape.finish(
                color=fitz.utils.getColor(color),
                fill_opacity=0.2,
                fill=fitz.utils.getColor(color),
                width=0,
            )
            if is_correct is not None:
                correct_color = "green" if is_correct else "red"
                shape.draw_line(
                    line.rect.top_left * derotation_matrix,
                    line.rect.bottom_left * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )

        if interval:
            # background color for depth interval
            background_rect = interval.background_rect
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
                if depth_is_correct is not None:
                    depth_is_correct_color = "green" if depth_is_correct else "red"
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
            line_anchor = interval.line_anchor
            if line_anchor:
                shape.draw_line(
                    line_anchor * derotation_matrix,
                    fitz.Point(layer_rect.x0, (layer_rect.y0 + layer_rect.y1) / 2) * derotation_matrix,
                )
                shape.finish(
                    color=fitz.utils.getColor(color),
                )
