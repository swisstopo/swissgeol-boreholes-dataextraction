"""Contains utility functions for plotting stratigraphic data."""

import logging

import cv2
import numpy as np
import pymupdf
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.textblock import TextBlock

logger = logging.getLogger(__name__)


def _draw_lines(open_cv_img, lines, scale_factor=1):
    grid_lines = [_convert_line_to_grid(line, scale_factor=scale_factor) for line in lines]
    for line_count, line in enumerate(grid_lines):
        color = (
            (255 - 5 * line_count) % 255,
            (5 * line_count) % 255,
            (10 * line_count) % 255,
        )
        try:
            cv2.line(
                open_cv_img,
                line.start.tuple,
                line.end.tuple,
                color,
                1,
            )
            cv2.circle(open_cv_img, line.start.tuple, radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(open_cv_img, line.end.tuple, radius=1, color=(0, 0, 255), thickness=-1)

        except cv2.error as e:
            logging.warning(f"Error drawing line. Exception: {e}. Skipping to draw the line.")

    return open_cv_img


def convert_page_to_opencv_img(page: pymupdf.Page, scale_factor: float, color_mode=cv2.COLOR_RGB2BGR) -> np.array:
    """Converts a pymupdf.Page object to an OpenCV image.

    Args:
        page (pymupdf.Page): The page to convert to an OpenCV image.
        scale_factor (float): Applied scale factor to the image.
        color_mode (_type_, optional): _description_. Defaults to cv2.COLOR_RGB2BGR.

    Returns:
        np.array: The OpenCV image.
    """
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    open_cv_img = cv2.cvtColor(img, color_mode)
    return open_cv_img


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


def plot_lines(
    page: pymupdf.Page, non_vertical_lines: list[Line], vertical_lines: list[Line] = None, scale_factor: float = 2
) -> cv2.COLOR_RGB2BGR:
    """Given a page object and the lines detected in the page, plot the page with the detected lines.

    Args:
        page (pymupdf.Page): The page to draw the lines in.
        non_vertical_lines (list[Line]): The non-vertical lines to plot.
        vertical_lines (list[Line], optional): The vertical lines to plot. Defaults to None.
        scale_factor (float, optional): The scale factor to apply to the pdf. Defaults to 2.

    Returns:
        open_cv_img: The page image with the lines drawn on it.
    """
    open_cv_img = convert_page_to_opencv_img(page, scale_factor=scale_factor)

    open_cv_img = _draw_lines(open_cv_img, non_vertical_lines, scale_factor=scale_factor)

    if vertical_lines:
        open_cv_img = _draw_lines(open_cv_img, vertical_lines, scale_factor=scale_factor)

    return open_cv_img


def draw_blocks_and_lines(page: pymupdf.Page, blocks: list[TextBlock], lines: list[Line] = None):
    """Draw the blocks and lines on the page.

    Args:
        page (pymupdf.Page): The page to draw the blocks and lines on.
        blocks (List[TextBlock]): The blocks to draw on the page.
        lines (List[Line] | None): The lines to draw on the page. Defaults to None.

    Returns:
        Union[cv2.COLOR_RGB2BGR, ArrayLike]: The image with the blocks and lines drawn on it.
    """
    scale_factor = 2

    for block in blocks:  # draw all blocks in the page
        pymupdf.utils.draw_rect(
            page,
            block.rect() * page.derotation_matrix,
            color=pymupdf.utils.getColor("orange"),
        )

    open_cv_img = convert_page_to_opencv_img(page, scale_factor=2)

    if lines is not None:
        open_cv_img = _draw_lines(open_cv_img, lines, scale_factor=scale_factor)

    return open_cv_img
