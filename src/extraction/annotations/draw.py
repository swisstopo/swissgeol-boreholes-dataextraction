"""This module contains functionalities to draw on pdf pages."""

import logging
import os
from pathlib import Path

import pandas as pd
import pymupdf
from dotenv import load_dotenv

from extraction.annotations.plot_utils import convert_page_to_opencv_img
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.stratigraphy.layer.layer import Layer
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from extraction.features.utils.table_detection import TableStructure

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
        filename = file_prediction.file_name
        logger.info("Drawing predictions for file %s", filename)
        try:
            # Clear cache to avoid cache contamination across different files, which can cause incorrect
            # visualizations; see also https://github.com/swisstopo/swissgeol-boreholes-suite/issues/1935
            pymupdf.TOOLS.store_shrink(100)

            with pymupdf.Document(directory / filename) as doc:
                for page_index, page in enumerate(doc):
                    page_number = page_index + 1
                    shape = page.new_shape()  # Create a shape object for drawing

                    # iterate over all boreholes identified
                    for borehole_predictions in file_prediction.borehole_predictions_list:
                        bounding_boxes = borehole_predictions.bounding_boxes
                        coordinates = borehole_predictions.metadata.coordinates
                        elevation = borehole_predictions.metadata.elevation
                        groundwaters = borehole_predictions.groundwater_in_borehole
                        bh_layers = borehole_predictions.layers_in_borehole

                        if coordinates is not None and page_number == coordinates.page_number:
                            draw_feature(
                                shape,
                                coordinates.rect * page.derotation_matrix,
                                coordinates.feature.is_correct,
                                "purple",
                            )
                        if elevation is not None and page_number == elevation.page_number:
                            draw_feature(
                                shape, elevation.rect * page.derotation_matrix, elevation.feature.is_correct, "blue"
                            )
                        for groundwater_entry in groundwaters.groundwater_feature_list:
                            if page_number == groundwater_entry.page_number:
                                draw_feature(
                                    shape,
                                    groundwater_entry.rect * page.derotation_matrix,
                                    groundwater_entry.feature.is_correct,
                                    "pink",
                                )
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
                                for layer in bh_layers.layers
                                if layer.material_description.page_number == page_number
                            ],
                        )

                    shape.commit()  # Commit all the drawing operations to the page

                    tmp_file_path = out_directory / f"{filename}_page{page_number}.png"
                    pymupdf.utils.get_pixmap(page, matrix=pymupdf.Matrix(2, 2), clip=page.rect).save(tmp_file_path)

                    if mlflow_tracking:  # This is only executed if MLFlow tracking is enabled
                        try:
                            import mlflow

                            mlflow.log_artifact(tmp_file_path, artifact_path="pages")
                        except NameError:
                            logger.warning("MLFlow could not be imported. Skipping logging of artifact.")

        except (FileNotFoundError, pymupdf.FileDataError) as e:
            logger.error("Error opening file %s: %s", filename, e)
            continue

        logger.info("Finished drawing predictions for file %s", filename)


def draw_feature(shape: pymupdf.Shape, rect: pymupdf.Rect, is_correct: bool | None, color: str) -> None:
    """Draw a bounding box around the area of the page where some feature was extracted from.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        rect (pymupdf:Rect): The bounding box of the feature to draw.
        is_correct (bool | None): whether the feature has been evaluated as correct against the ground truth
        color (str): The name of the color to use for the bounding box of the feature.
    """
    shape.draw_rect(rect)
    shape.finish(color=pymupdf.utils.getColor(color))

    if is_correct is not None:
        correct_color = "green" if is_correct else "red"
        shape.draw_line(
            rect.top_left,
            rect.bottom_left,
        )
        shape.finish(
            color=pymupdf.utils.getColor(correct_color),
            width=6,
            stroke_opacity=0.5,
        )


def draw_material_descriptions(shape: pymupdf.Shape, derotation_matrix: pymupdf.Matrix, layers: list[Layer]) -> None:
    """Draw information about material descriptions on a pdf page.

    In particular, this function:
        - draws rectangles around the material description text blocks,
        - draws lines connecting the material description text blocks to the depth intervals,
        - colors the lines of the material description text blocks based on whether they were correctly identified.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        derotation_matrix (pymupdf.Matrix): The derotation matrix of the page.
        layers (LayerPrediction): The predictions for the page.
    """
    for index, layer in enumerate(layers):
        if layer.material_description.rect is not None:
            shape.draw_rect(
                pymupdf.Rect(layer.material_description.rect) * derotation_matrix,
            )
            shape.finish(color=pymupdf.utils.getColor("orange"))
        draw_layer(shape=shape, derotation_matrix=derotation_matrix, layer=layer, index=index)


def draw_depth_columns_and_material_rect(
    shape: pymupdf.Shape, derotation_matrix: pymupdf.Matrix, bounding_boxes: list[PageBoundingBoxes]
):
    """Draw depth columns as well as the material rects on a pdf page.

    In particular, this function:
        - draws rectangles around the depth columns,
        - draws rectangles around the depth column entries,
        - draws rectangles around the material description columns.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        derotation_matrix (pymupdf.Matrix): The derotation matrix of the page.
        bounding_boxes (list[BoundingBoxes]): List of bounding boxes for depth column and material descriptions.
    """
    for bboxes in bounding_boxes:
        if bboxes.sidebar_bbox:  # Draw rectangle for depth columns
            shape.draw_rect(
                pymupdf.Rect(bboxes.sidebar_bbox.rect) * derotation_matrix,
            )
            shape.finish(color=pymupdf.utils.getColor("green"))
            for depth_column_entry in bboxes.depth_column_entry_bboxes:  # Draw rectangle for depth column entries
                shape.draw_rect(
                    pymupdf.Rect(depth_column_entry.rect) * derotation_matrix,
                )
            shape.finish(color=pymupdf.utils.getColor("purple"))

        shape.draw_rect(  # Draw rectangle for material description column
            bboxes.material_description_bbox.rect * derotation_matrix,
        )
        shape.finish(color=pymupdf.utils.getColor("red"))


def draw_layer(shape: pymupdf.Shape, derotation_matrix: pymupdf.Matrix, layer: Layer, index: int):
    """Draw layers on a pdf page.

    In particular, this function:
        - draws lines connecting the material description text layers to the depth intervals,
        - colors the lines of the material description text layers based on whether they were correctly identified.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        derotation_matrix (pymupdf.Matrix): The derotation matrix of the page.
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
                color=pymupdf.utils.getColor(color),
                fill_opacity=0.2,
                fill=pymupdf.utils.getColor(color),
                width=0,
            )
            if material_description.is_correct is not None:
                correct_color = "green" if material_description.is_correct else "red"
                shape.draw_line(
                    line.rect.top_left * derotation_matrix,
                    line.rect.bottom_left * derotation_matrix,
                )
                shape.finish(
                    color=pymupdf.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )

        if layer.depths:
            # line from depth interval to material description
            depths_rect = pymupdf.Rect()
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
                        pymupdf.Point(rect.x0, (rect.y0 + rect.y1) / 2) * derotation_matrix,
                    )
                    shape.finish(
                        color=pymupdf.utils.getColor(color),
                    )

                # background color for depth interval
                background_rect = layer.depths.background_rect
                if background_rect is not None:
                    shape.draw_rect(
                        background_rect * derotation_matrix,
                    )
                    shape.finish(
                        color=pymupdf.utils.getColor(color),
                        fill_opacity=0.2,
                        fill=pymupdf.utils.getColor(color),
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
                            color=pymupdf.utils.getColor(depth_is_correct_color),
                            width=6,
                            stroke_opacity=0.5,
                        )
            else:
                # Depths are part of the material description: only draw bounding boxes
                if layer.depths.start:
                    shape.draw_rect(pymupdf.Rect(layer.depths.start.rect) * derotation_matrix)
                if layer.depths.end:
                    shape.draw_rect(pymupdf.Rect(layer.depths.end.rect) * derotation_matrix)
                shape.finish(color=pymupdf.utils.getColor("purple"))

                # draw green line if depth interval is correct else red line
                if layer.is_correct is not None:
                    depth_is_correct_color = "green" if layer.is_correct else "red"
                    if layer.depths.start:
                        shape.draw_line(
                            layer.depths.start.rect.bottom_left * derotation_matrix,
                            layer.depths.start.rect.bottom_right * derotation_matrix,
                        )
                    if layer.depths.end:
                        shape.draw_line(
                            layer.depths.end.rect.bottom_left * derotation_matrix,
                            layer.depths.end.rect.bottom_right * derotation_matrix,
                        )
                    shape.finish(
                        color=pymupdf.utils.getColor(depth_is_correct_color),
                        width=6,
                        stroke_opacity=0.5,
                    )


def draw_table_structures(
    shape: pymupdf.Shape, derotation_matrix: pymupdf.Matrix, tables: list[TableStructure]
) -> None:
    """Draw table structures on a pdf page.

    If multiple tables are available each is drawn in a different color to distinguish between the detected tables.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        derotation_matrix (pymupdf.Matrix): The derotation matrix of the page.
        tables (list[TableStructure]): List of detected table structures.
    """
    # Define colors for different tables
    table_colors = [
        ("purple", "mediumpurple"),
        ("blue", "lightblue"),
        ("green", "lightgreen"),
        ("red", "lightcoral"),
        ("orange", "peachpuff"),
        ("brown", "tan"),
        ("darkgreen", "lightseagreen"),
        ("darkblue", "lightsteelblue"),
    ]

    for index, table in enumerate(tables):
        main_color, light_color = table_colors[index % len(table_colors)]

        # Draw the table bounding rectangle
        shape.draw_rect(table.bounding_rect * derotation_matrix)
        shape.finish(color=pymupdf.utils.getColor(main_color), width=3, stroke_opacity=0.8)

        # Draw horizontal and vertical lines in a lighter shade
        for h_line in table.horizontal_lines:
            start_point = pymupdf.Point(h_line.start.x, h_line.start.y)
            end_point = pymupdf.Point(h_line.end.x, h_line.end.y)
            shape.draw_line(start_point * derotation_matrix, end_point * derotation_matrix)

        shape.finish(color=pymupdf.utils.getColor(light_color), width=2, stroke_opacity=0.6)

        for v_line in table.vertical_lines:
            start_point = pymupdf.Point(v_line.start.x, v_line.start.y)
            end_point = pymupdf.Point(v_line.end.x, v_line.end.y)
            shape.draw_line(start_point * derotation_matrix, end_point * derotation_matrix)

        shape.finish(color=pymupdf.utils.getColor(light_color), width=1, stroke_opacity=0.6)


def plot_tables(page: pymupdf.Page, table_structures: list[TableStructure], page_index: int):
    """Draw table structures on a pdf page.

    Args:
        page:               The PDF page.
        table_structures:   The identified table structures on the page.
        page_index:         The index of the page in the document (0-based).
    """
    temp_doc = pymupdf.open()
    temp_doc.insert_pdf(page.parent, from_page=page_index, to_page=page_index)
    temp_page = temp_doc[0]

    shape = temp_page.new_shape()
    draw_table_structures(shape, temp_page.derotation_matrix, table_structures)
    shape.commit()

    result = convert_page_to_opencv_img(temp_page, scale_factor=2)

    temp_doc.close()

    return result
