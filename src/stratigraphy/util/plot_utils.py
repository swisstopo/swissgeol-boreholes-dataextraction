"""Contains utility functions for plotting stratigraphic data."""

import cv2
import fitz
import numpy as np

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.textblock import TextBlock


def _draw_lines(open_cv_img, lines, scale_factor=1):
    for line_count, line in enumerate(lines):
        color = (
            (255 - 5 * line_count) % 255,
            (5 * line_count) % 255,
            (10 * line_count) % 255,
        )
        try:
            cv2.line(
                open_cv_img,
                np.dot(scale_factor, line.start.tuple),
                np.dot(scale_factor, line.end.tuple),
                color,
                1,
            )
            cv2.circle(open_cv_img, np.dot(scale_factor, line.start.tuple), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(open_cv_img, np.dot(scale_factor, line.end.tuple), radius=1, color=(0, 0, 255), thickness=-1)

        except cv2.error as e:
            print("Error drawing line")
            print(f"Line points: {line.start.tuple}, {line.end.tuple}")
            print(e)
            print()
    return open_cv_img


def _convert_page_to_opencv_img(page, scale_factor):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    open_cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return open_cv_img


def plot_lines(page: fitz.Page, lines: list[Line], scale_factor: float = 2) -> cv2.COLOR_RGB2BGR:
    """Given a page object and the lines detected in the page, plot the page with the detected lines.

    Args:
        page (fitz.Page): The page to draw the lines in.
        lines (ArrayLike): The lines detected in the pdf.
        scale_factor (float, optional): The scale factor to apply to the pdf. Defaults to 2.
    """
    open_cv_img = _convert_page_to_opencv_img(page, scale_factor=scale_factor)

    open_cv_img = _draw_lines(open_cv_img, lines, scale_factor=scale_factor)

    return open_cv_img


def draw_blocks_and_lines(page: fitz.Page, blocks: list[TextBlock], lines: list[Line] = None):
    """Draw the blocks and lines on the page.

    Args:
        page (fitz.Page): The page to draw the blocks and lines on.
        blocks (List[TextBlock]): The blocks to draw on the page.
        lines (List[Line] | None): The lines to draw on the page. Defaults to None.

    Returns:
        Union[cv2.COLOR_RGB2BGR, ArrayLike]: The image with the blocks and lines drawn on it.
    """
    scale_factor = 2

    for block in blocks:  # draw all blocks in the page
        fitz.utils.draw_rect(
            page,
            block.rect() * page.derotation_matrix,
            color=fitz.utils.getColor("orange"),
        )

    open_cv_img = _convert_page_to_opencv_img(page, scale_factor=2)

    if lines is not None:
        open_cv_img = _draw_lines(open_cv_img, lines, scale_factor=scale_factor)

    return open_cv_img
