"""This module contains functionalities to draw on pdf pages."""

import logging
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

from extraction.annotations.plot_utils import convert_page_to_opencv_img
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.stratigraphy.layer.layer import Layer
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from extraction.features.utils.strip_log_detection import StripLog
from extraction.features.utils.table_detection import StructureLine, TableStructure

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

colors = ["purple", "blue"]

logger = logging.getLogger(__name__)


def draw_predictions(
    predictions: OverallFilePredictions,
    directory: Path,
    out_directory: Path,
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
                for _, page in enumerate(doc):
                    drawer = PageDrawer(page)
                    drawer.draw(file_prediction)

                    tmp_file_path = out_directory / f"{filename}_page{drawer.page_number}.png"
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


class PageDrawer:
    """Helper class for drawing predictions on a page."""

    def __init__(self, page: pymupdf.Page):
        """Creates a PageDrawer instance from a PDF page.

        Args:
            page: the PDF page to draw predictions on
        """
        self.page = page
        self.page_number = page.number + 1
        self.shape = page.new_shape()  # Create a shape object for drawing

    def draw(self, file_predictions: FilePredictions):
        """Draws extracted borehole data on the page.

        Args:
            file_predictions: the extracted borehole data
        """
        # iterate over all boreholes identified
        for borehole_predictions in file_predictions.borehole_predictions_list:
            bounding_boxes = borehole_predictions.bounding_boxes
            coordinates = borehole_predictions.metadata.coordinates
            elevation = borehole_predictions.metadata.elevation
            groundwaters = borehole_predictions.groundwater_in_borehole
            bh_layers = borehole_predictions.layers_in_borehole

            if coordinates is not None and self.page_number == coordinates.page_number:
                self.draw_feature(
                    coordinates.rect * self.page.derotation_matrix,
                    coordinates.feature.is_correct,
                    "purple",
                )
            if elevation is not None and self.page_number == elevation.page_number:
                self.draw_feature(elevation.rect * self.page.derotation_matrix, elevation.feature.is_correct, "blue")
            for groundwater_entry in groundwaters.groundwater_feature_list:
                if self.page_number == groundwater_entry.page_number:
                    self.draw_feature(
                        groundwater_entry.rect * self.page.derotation_matrix,
                        groundwater_entry.feature.is_correct,
                        "pink",
                    )
            page_bounding_boxes = [bboxes for bboxes in bounding_boxes if bboxes.page == self.page_number]
            for bboxes in page_bounding_boxes:
                self.draw_bounding_boxes(bboxes)

            layers = [layer for layer in bh_layers.layers if self.page_number in layer.material_description.pages]
            for index, layer in enumerate(layers):
                self.draw_layer(
                    layer=layer,
                    sidebar_rects=[bboxes.sidebar_bbox.rect for bboxes in page_bounding_boxes if bboxes.sidebar_bbox],
                    index=index,
                    page_number=self.page_number,
                )

        self.shape.commit()  # Commit all the drawing operations to the page

    def draw_feature(self, rect: pymupdf.Rect, is_correct: bool | None, color: str) -> None:
        """Draw a bounding box around the area of the page where some feature was extracted from.

        Args:
            rect (pymupdf:Rect): The bounding box of the feature to draw.
            is_correct (bool | None): whether the feature has been evaluated as correct against the ground truth
            color (str): The name of the color to use for the bounding box of the feature.
        """
        self.shape.draw_rect(rect)
        self.shape.finish(color=pymupdf.utils.getColor(color))

        self._draw_correctness_line(start=rect.top_left, end=rect.bottom_left, is_correct=is_correct, width=6)

    def draw_bounding_boxes(self, bounding_boxes: PageBoundingBoxes):
        """Draw depth columns as well as the material rects on a pdf page.

        In particular, this function:
            - draws rectangles around the depth columns,
            - draws rectangles around the depth column entries,
            - draws rectangles around the material description columns.

        Args:
            bounding_boxes (BoundingBoxes): bounding boxes for depth column and material descriptions.
        """
        if bounding_boxes.sidebar_bbox:  # Draw rectangle for depth columns
            self.shape.draw_rect(
                pymupdf.Rect(bounding_boxes.sidebar_bbox.rect) * self.page.derotation_matrix,
            )
            self.shape.finish(color=pymupdf.utils.getColor("green"))
            for (
                depth_column_entry
            ) in bounding_boxes.depth_column_entry_bboxes:  # Draw rectangle for depth column entries
                self.shape.draw_rect(
                    pymupdf.Rect(depth_column_entry.rect) * self.page.derotation_matrix,
                )
            self.shape.finish(color=pymupdf.utils.getColor("purple"))

        self.shape.draw_rect(  # Draw rectangle for material description column
            bounding_boxes.material_description_bbox.rect * self.page.derotation_matrix,
        )
        self.shape.finish(color=pymupdf.utils.getColor("red"))

    def draw_layer(
        self,
        layer: Layer,
        sidebar_rects: list[pymupdf.Rect],
        index: int,
        page_number: int,
    ):
        """Draw layers on a pdf page.

        In particular, this function:
            - draws lines connecting the material description text layers to the depth intervals,
            - colors the lines of the material description text layers based on whether they were correctly identified.

        Args:
            layer (Layer): The layer (depth interval and material description).
            sidebar_rects (list[pymupdf.Rect]): The sidebar bounding boxes on the page.
            index (int): Index of the layer.
            page_number(int): the curent page number.
        """
        material_description = layer.material_description
        mat_descr_lines = [line for line in material_description.lines if line.page_number == page_number]
        mat_descr_rect = layer.material_description.rect_for_page(page_number)
        if mat_descr_lines:
            color = colors[index % len(colors)]

            # background color for material description
            for line in mat_descr_lines:
                self.shape.draw_rect(line.rect * self.page.derotation_matrix)
                self.shape.finish(
                    color=pymupdf.utils.getColor(color),
                    fill_opacity=0.2,
                    fill=pymupdf.utils.getColor(color),
                    width=0,
                )
                self._draw_correctness_line(
                    start=pymupdf.Point(line.rect.top_left.x - 6, line.rect.top_left.y),
                    end=pymupdf.Point(line.rect.bottom_left.x - 6, line.rect.bottom_left.y),
                    is_correct=layer.is_correct,
                    width=6,
                )
                self._draw_correctness_line(
                    start=line.rect.top_left,
                    end=line.rect.bottom_left,
                    is_correct=material_description.is_correct,
                    width=4,
                )

            if layer.depths:
                depths_rect = pymupdf.Rect()
                if layer.depths.start and layer.depths.start.rect and layer.depths.start.page_number == page_number:
                    depths_rect.include_rect(layer.depths.start.rect)
                if layer.depths.end and layer.depths.end.page_number == page_number:
                    depths_rect.include_rect(layer.depths.end.rect)

                if any(sidebar_rect.contains(depths_rect) for sidebar_rect in sidebar_rects):
                    # Depths are separate from the material description: draw a line connecting them
                    line_anchor = layer.depths.get_line_anchor(page_number)
                    if line_anchor:
                        rect = mat_descr_rect
                        self.shape.draw_line(
                            line_anchor * self.page.derotation_matrix,
                            pymupdf.Point(rect.x0, (rect.y0 + rect.y1) / 2) * self.page.derotation_matrix,
                        )
                        self.shape.finish(
                            color=pymupdf.utils.getColor(color),
                        )

                    # background color for depth interval
                    background_rect = layer.depths.get_background_rect(page_number, self.shape.page.rect.height)
                    if background_rect is not None:
                        self.shape.draw_rect(
                            background_rect * self.page.derotation_matrix,
                        )
                        self.shape.finish(
                            color=pymupdf.utils.getColor(color),
                            fill_opacity=0.2,
                            fill=pymupdf.utils.getColor(color),
                            width=0,
                        )

                        self._draw_correctness_line(
                            start=background_rect.top_left,
                            end=background_rect.bottom_left,
                            is_correct=layer.depths.is_correct,
                            width=4,
                        )
                else:
                    # Depths are extracted from the material description: only draw bounding boxes
                    if (
                        layer.depths.start
                        and layer.depths.start.rect
                        and layer.depths.start.page_number == page_number
                    ):
                        self.shape.draw_rect(pymupdf.Rect(layer.depths.start.rect) * self.page.derotation_matrix)
                    if layer.depths.end and layer.depths.end.page_number == page_number:
                        self.shape.draw_rect(pymupdf.Rect(layer.depths.end.rect) * self.page.derotation_matrix)
                    self.shape.finish(color=pymupdf.utils.getColor("purple"))

                    if (
                        layer.depths.start
                        and layer.depths.start.rect
                        and layer.depths.start.page_number == page_number
                    ):
                        self._draw_correctness_line(
                            start=layer.depths.start.rect.bottom_left,
                            end=layer.depths.start.rect.bottom_right,
                            is_correct=layer.depths.is_correct,
                            width=4,
                        )
                    if layer.depths.end and layer.depths.end.page_number == page_number:
                        self._draw_correctness_line(
                            start=layer.depths.end.rect.bottom_left,
                            end=layer.depths.end.rect.bottom_right,
                            is_correct=layer.depths.is_correct,
                            width=4,
                        )

    def _draw_correctness_line(
        self,
        start: pymupdf.Point,
        end: pymupdf.Point,
        is_correct: bool | None,
        width: int,
    ) -> None:
        if is_correct is not None:
            depth_is_correct_color = "green" if is_correct else "red"
            self.shape.draw_line(
                start * self.page.derotation_matrix,
                end * self.page.derotation_matrix,
            )
            self.shape.finish(
                color=pymupdf.utils.getColor(depth_is_correct_color),
                width=width,
                stroke_opacity=0.5,
            )


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


def draw_strip_logs(shape: pymupdf.Shape, derotation_matrix: pymupdf.Matrix, strip_logs: list[StripLog]) -> None:
    """Draw strip log structures on a pdf page.

    Each strip log is drawn with a distinctive color to distinguish between different detected strip logs.

    Args:
        shape (pymupdf.Shape): The shape object for drawing.
        derotation_matrix (pymupdf.Matrix): The derotation matrix of the page.
        strip_logs (list[StripLog]): List of detected strip log structures.
    """
    # Define colors for different strip logs
    strip_colors = [
        "blue",
        "green",
        "red",
        "orange",
        "brown",
        "purple",
        "darkgreen",
        "darkblue",
    ]

    for index, strip in enumerate(strip_logs):
        main_color = strip_colors[index % len(strip_colors)]

        # Draw the striplog bounding rectangle with thick border
        shape.draw_rect(strip.bbox * derotation_matrix)
        shape.finish(color=pymupdf.utils.getColor(main_color), width=4, stroke_opacity=0.9)

        # Draw subsections with light border
        for section in strip.sections:
            shape.draw_rect(section.bbox * derotation_matrix)
            shape.finish(color=pymupdf.utils.getColor(main_color), width=1, stroke_opacity=0.3)


def plot_strip_logs(page: pymupdf.Page, strip_logs: list[StripLog], page_index: int):
    """Draw strip log structures on a pdf page.

    Args:
        page: The PDF page.
        strip_logs: The identified strip log structures on the page.
        page_index: The index of the page in the document (0-based).
    """
    temp_doc = pymupdf.open()
    temp_doc.insert_pdf(page.parent, from_page=page_index, to_page=page_index)
    temp_page = temp_doc[0]

    shape = temp_page.new_shape()
    draw_strip_logs(shape, temp_page.derotation_matrix, strip_logs)
    shape.commit()

    result = convert_page_to_opencv_img(temp_page, scale_factor=2)

    temp_doc.close()

    return result
