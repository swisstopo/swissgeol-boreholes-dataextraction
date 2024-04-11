"""This module contains functionalities to draw on pdf pages."""

import logging
import os
from pathlib import Path

import fitz
from dotenv import load_dotenv

from stratigraphy.util.interval import BoundaryInterval
from stratigraphy.util.predictions import FilePredictions, LayerPrediction
from stratigraphy.util.textblock import TextBlock

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

colors = ["purple", "blue"]

logger = logging.getLogger(__name__)


def draw_predictions(predictions: list[FilePredictions], directory: Path, out_directory: Path) -> None:
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
    if directory.is_file():  # deal with the case when we pass a file instead of a directory
        directory = directory.parent
    for file_name, file_prediction in predictions.items():
        with fitz.Document(directory / file_name) as doc:
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                layers = file_prediction.pages[page_index].layers
                depths_materials_column_pairs = file_prediction.pages[page_index].depths_materials_columns_pairs
                draw_depth_columns_and_material_rect(page, depths_materials_column_pairs)
                draw_material_descriptions(page, layers)

                tmp_file_path = out_directory / f"{file_name}_page{page_number}.png"
                fitz.utils.get_pixmap(page, matrix=fitz.Matrix(2, 2), clip=page.rect).save(tmp_file_path)
                if mlflow_tracking:  # This is only executed if MLFlow tracking is enabled
                    try:
                        import mlflow

                        mlflow.log_artifact(tmp_file_path, artifact_path="pages")
                    except NameError:
                        logger.warning("MLFlow could not be imported. Skipping logging of artifact.")


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
    for index, layer in enumerate(layers):
        if layer.material_description.rect is not None:
            fitz.utils.draw_rect(
                page,
                fitz.Rect(layer.material_description.rect) * page.derotation_matrix,
                color=fitz.utils.getColor("orange"),
            )
        draw_layer(
            page=page,
            interval=layer.depth_interval,  # None if no depth interval
            layer=layer.material_description,
            index=index,
            is_correct=layer.material_is_correct,  # None if no ground truth
        )


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
    for pair in depths_materials_column_pairs:
        depth_column = pair["depth_column"]
        material_description_rect = pair["material_description_rect"]
        if depth_column is not None:  # draw rectangle for depth columns
            page.draw_rect(
                fitz.Rect(depth_column["rect"]) * page.derotation_matrix,
                color=fitz.utils.getColor("green"),
            )
            for depth_column_entry in depth_column["entries"]:  # draw rectangle for depth column entries
                fitz.utils.draw_rect(
                    page,
                    fitz.Rect(depth_column_entry["rect"]) * page.derotation_matrix,
                    color=fitz.utils.getColor("purple"),
                )

        fitz.utils.draw_rect(  # draw rectangle for material description column
            page,
            fitz.Rect(material_description_rect) * page.derotation_matrix,
            color=fitz.utils.getColor("red"),
        )

    return page


def draw_layer(
    page: fitz.Page,
    interval: BoundaryInterval | None,
    layer: TextBlock,
    index: int,
    is_correct: bool,
):
    """Draw layers on a pdf page.

    In particular, this function:
        - draws lines connecting the material description text layers to the depth intervals,
        - colors the lines of the material description text layers based on whether they were correctly identified.

    Args:
        page (fitz.Page): The page to draw on.
        interval (dict | None): Depth interval for the layer.
        layer (MaterialDescriptionPrediction): Material description block for the layer.
        index (int): Index of the layer.
        is_correct (bool): Whether the text block was correctly identified.
    """
    if len(layer.lines):
        layer_rect = fitz.Rect(layer.rect)
        color = colors[index % len(colors)]

        # background color for material description
        for line in [line for line in layer.lines]:
            page.draw_rect(
                line.rect * page.derotation_matrix, width=0, fill=fitz.utils.getColor(color), fill_opacity=0.2
            )
            if is_correct is not None:
                correct_color = "green" if is_correct else "red"
                page.draw_line(
                    line.rect.top_left * page.derotation_matrix,
                    line.rect.bottom_left * page.derotation_matrix,
                    color=fitz.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )

        if interval:
            # background color for depth interval
            # background_rect = _background_rect(interval)
            background_rect = interval.background_rect
            if background_rect is not None:
                page.draw_rect(
                    background_rect * page.derotation_matrix,
                    width=0,
                    fill=fitz.utils.getColor(color),
                    fill_opacity=0.2,
                )

            # line from depth interval to material description
            # line_anchor = _get_line_anchor(interval)
            line_anchor = interval.line_anchor
            if line_anchor:
                page.draw_line(
                    line_anchor * page.derotation_matrix,
                    fitz.Point(layer_rect.x0, (layer_rect.y0 + layer_rect.y1) / 2) * page.derotation_matrix,
                    color=fitz.utils.getColor(color),
                )
