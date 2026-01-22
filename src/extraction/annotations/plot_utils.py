"""Contains utility functions for plotting stratigraphic data."""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textblock import TextBlock

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow

logger = logging.getLogger(__name__)


def _get_centered_hline(center: pymupdf.Point, scale: int) -> tuple[pymupdf.Point, pymupdf.Point]:
    """Return a horizontal line centered on `center`.

    The line goes from (center.x - scale) to (center.x + scale) at the same y.

    Args:
        center (pymupdf.Point): Center point of the line.
        scale (int): Half-length of the line (distance from center to each end).

    Returns:
        tuple[pymupdf.Point, pymupdf.Point]: Points of the line.
    """
    return pymupdf.Point(center.x - scale, center.y), pymupdf.Point(center.x + scale, center.y)


def _get_polyline_triangle(
    center: pymupdf.Point, is_up: bool = False, width: int = 6, height: int = 6
) -> list[pymupdf.Point]:
    """Return a polyline (list of points) for a triangle.

    Args:
        center (pymupdf.Point): Center of the base of the triangle.
        is_up (bool, optional): If True, the tip is above the base; otherwise below.
        width (int, optional): Base width of the triangle. Defaults to 6.
        height (int, optional): Height of the triangle. Defaults to 6.

    Returns:
        list[pymupdf.Point]: List of points describing the triangle as a polyline.
    """
    return [
        pymupdf.Point(center.x - width // 2, center.y - ((-1) ** is_up) * height),
        pymupdf.Point(center.x + width // 2, center.y - ((-1) ** is_up) * height),
        pymupdf.Point(center.x, center.y),
        pymupdf.Point(center.x - width // 2, center.y - ((-1) ** is_up) * height),
    ]


def _draw_lines(img: np.ndarray, lines: list[Line], scale_factor: int = 1) -> np.ndarray:
    """Draw lines on image.

    Args:
        img (np.ndarray): Input image to draw on.
        lines (list[Line]): Lines to draw
        scale_factor (int, optional): Scaling factor. Defaults to 1.

    Returns:
        np.ndarray: Lines plotted on image.
    """
    grid_lines = [_convert_line_to_grid(line, scale_factor=scale_factor) for line in lines]
    img = img.copy()
    for line_count, line in enumerate(grid_lines):
        color = (
            (255 - 5 * line_count) % 255,
            (5 * line_count) % 255,
            (10 * line_count) % 255,
        )
        try:
            cv2.line(img, line.start.tuple, line.end.tuple, color=color, thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, line.start.tuple, radius=1, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, line.end.tuple, radius=1, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

        except cv2.error as e:
            logging.warning(f"Error drawing line. Exception: {e}. Skipping to draw the line.")

    return img


def convert_page_to_img(page: pymupdf.Page, scale_factor: float) -> np.ndarray:
    """Converts a pymupdf.Page object to an image.

    Args:
        page (pymupdf.Page): The page to convert to an image.
        scale_factor (float): Applied scale factor to the image.

    Returns:
        np.ndarray: The image.
    """
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return img


def _convert_line_to_grid(line: Line, scale_factor: float) -> Line:
    """Convert the line to a grid.

    Note: OpenCV uses a pixel grid system as a coordinate system, and as such only allows
    integer values for the coordinates. This function converts the lines to a grid by
    applying the right scale factor and then rounding the coordinates to the nearest integer.

    Args:
        line (Line): The line to convert to a grid.
        scale_factor (float): The scale factor to apply to the lines.

    Returns:
        Line: The lines converted to a grid.
    """
    start = line.start
    start.x = int(np.round(scale_factor * start.x, 0))
    start.y = int(np.round(scale_factor * start.y, 0))
    end = line.end
    end.x = int(np.round(scale_factor * end.x, 0))
    end.y = int(np.round(scale_factor * end.y, 0))
    return Line(start, end)


def plot_lines(page: pymupdf.Page, geometric_lines: list[Line], scale_factor: float = 2) -> np.ndarray:
    """Given a page object and the lines detected in the page, plot the page with the detected lines.

    Args:
        page (pymupdf.Page): The page to draw the lines in.
        geometric_lines (list[Line]): The geometric_lines lines to plot.
        scale_factor (float, optional): The scale factor to apply to the pdf. Defaults to 2.

    Returns:
        img (np.ndarray): The page image with the lines drawn on it.
    """
    img = convert_page_to_img(page, scale_factor=scale_factor)
    img = _draw_lines(img, geometric_lines, scale_factor=scale_factor)
    return img


def draw_blocks_and_lines(
    page: pymupdf.Page, blocks: list[TextBlock], lines: list[Line] = None, scale_factor: int = 2
):
    """Draw the blocks and lines on the page.

    Args:
        page (pymupdf.Page): The page to draw the blocks and lines on.
        blocks (List[TextBlock]): The blocks to draw on the page.
        lines (List[Line] | None): The lines to draw on the page. Defaults to None.
        scale_factor (int): Scaling factor for image.

    Returns:
        Union[cv2.COLOR_RGB2BGR, ArrayLike]: The image with the blocks and lines drawn on it.
    """
    for block in blocks:  # draw all blocks in the page
        pymupdf.utils.draw_rect(
            page,
            block.rect() * page.derotation_matrix,
            color=pymupdf.utils.getColor("orange"),
        )

    open_cv_img = convert_page_to_img(page, scale_factor=scale_factor)

    if lines is not None:
        open_cv_img = _draw_lines(open_cv_img, lines, scale_factor=scale_factor)

    return open_cv_img


def save_visualization(img, filename, page_number, visualization_type, draw_directory, mlflow_tracking):
    """Save visualization image to file and/or MLflow."""
    if draw_directory:
        img_path = draw_directory / f"{Path(filename).stem}_page_{page_number}_{visualization_type}.png"
        cv2.imwrite(str(img_path), img)

    if mlflow_tracking:
        mlflow.log_image(img, f"pages/{filename}_page_{page_number}_{visualization_type}.png")

    elif not draw_directory:
        logger.warning(f"draw_directory is not defined. Skipping saving {visualization_type} image.")
