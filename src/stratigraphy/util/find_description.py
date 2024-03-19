"""This module contains functions to find the description (blocks) of a material in a pdf page."""

from collections.abc import Callable

import fitz
import numpy as np

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.line import TextLine
from stratigraphy.util.textblock import TextBlock


def get_description_lines(lines: list[TextLine], material_description_rect: fitz.Rect) -> list[TextLine]:
    """Get the description lines of a material.

    Checks if the lines are within the material description rectangle and if they are not too far to the right.

    Args:
        lines (list[TextLine]): The lines to filter.
        material_description_rect (fitz.Rect): The rectangle containing the material description.

    Returns:
        list[TextLine]: The filtered lines.
    """
    filtered_lines = [
        line
        for line in lines
        if line.rect.x0 < material_description_rect.x1 - 0.4 * material_description_rect.width
        if material_description_rect.contains(line.rect)
    ]

    return sorted([line for line in filtered_lines if line], key=lambda line: line.rect.y0)


def get_description_blocks(
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    block_line_ratio: float,
    left_line_length_threshold: float,
    target_layer_count: int = None,
) -> list[TextBlock]:
    """Group the description lines into blocks.

    The grouping is done based on the presence of geometric lines, the indentation of lines
    and the vertical spacing between lines.

    Args:
        description_lines (list[TextLine]): The text lines to group into blocks.
        geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        block_line_ratio (float): The relative part a line has to cross a block in order to induce a splitting.
        left_line_length_threshold (float): The minimum length of a line segment on the left side
                                            of a block to split it.
        target_layer_count (int, optional): Expected number of blocks. Defaults to None.

    Returns:
        list[TextBlock]: A list of blocks containing the description lines.
    """
    distances = []
    for line_index in range(len(description_lines) - 1):
        line1rect = description_lines[line_index].rect
        line2rect = description_lines[line_index + 1].rect
        if line2rect.y0 > line1rect.y0 + line1rect.height / 2:
            distances.append(line2rect.y0 - line1rect.y0)

    threshold = None
    if len(distances):
        threshold = min(distances) * 1.15

    # Create blocks separated by lines
    blocks = _create_blocks_by_separator(
        description_lines,
        geometric_lines,
        _block_separated_by_line,
        {"threshold": block_line_ratio},
        set_terminated_by_line_flag=True,
    )

    # Create blocks separated by lefthandside line segments
    _blocks = []
    for block in blocks:
        _blocks.extend(
            _create_blocks_by_separator(
                block.lines,
                geometric_lines,
                _block_separated_by_lefthandside_line_segment,
                {"length_threshold": left_line_length_threshold},
                set_terminated_by_line_flag=False,
            )
        )
        if block.is_terminated_by_line:  # keep the line termination if it was there
            _blocks[-1].is_terminated_by_line = True
    blocks = _blocks

    min_block_count = 3 if target_layer_count is None else 2 / 3 * target_layer_count
    # If we have only found one splitting line, then we fall back to considering vertical spacing, as it is more
    # likely that this line is a false positive, than that we have a borehole profile with only two layers.
    # If the number of blocks is less than 2/3 of the expected number of layers (based on the information from the
    # depth column, then the splitting based on horizontal lines is not reliable, and we fall back to considering
    # vertical spacing between text.

    count_blocks_divided_by_line = len([block for block in blocks if block.is_terminated_by_line])
    if len(blocks) < min_block_count:
        # This case means that there are fewer blocks than the minimum number of blocks we expect.
        # In this case we redo all the blocks from scratch.
        blocks = _split_block_by_vertical_spacing(description_lines, threshold=threshold)

    elif count_blocks_divided_by_line < min_block_count:
        # In this case there blocks due to line segments. However, they are mostly due to small segments
        # on the lefthandside of the blocks. Minimum there are fewer blocks due to lines than min_block_count.
        # Often, these lefthandside lines are only used when space is tight. If space is not tight, those
        # indicators are dropped. That's why we have to consider vertical spacing as well.
        _blocks = []
        for block in blocks:
            _blocks.extend(_split_block_by_vertical_spacing(block.lines, threshold=threshold))
        blocks = _blocks
    blocks = [new_block for block in blocks for new_block in block.split_based_on_indentation()]

    return blocks


def _create_blocks_by_separator(
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    separator_condition: Callable,
    separator_params,
    set_terminated_by_line_flag: bool,
) -> list[TextBlock]:
    blocks = []
    current_block_lines = []
    for line in description_lines:
        if len(current_block_lines) > 0:
            last_line = current_block_lines[-1]
            if separator_condition(
                last_line, line, TextBlock(current_block_lines), geometric_lines, **separator_params
            ):
                blocks.append(TextBlock(current_block_lines, is_terminated_by_line=set_terminated_by_line_flag))
                current_block_lines = []
        current_block_lines.append(line)
    if len(current_block_lines):
        blocks.append(TextBlock(current_block_lines))
    return blocks


def _split_block_by_vertical_spacing(description_lines: list[TextLine], threshold: int) -> list[TextBlock]:
    """Split the description lines into blocks based on the vertical spacing between the text lines.

    Args:
        description_lines (list[TextLine]): The text lines to split into blocks.
        threshold (int): The maximum vertical distance between two lines to be considered part of the same block.

    Returns:
        list[TextBlock]: List of blocks.
    """
    # No good splitting by lines found, so create blocks by looking at vertical spacing instead
    blocks = []
    current_block_lines = []
    for line in description_lines:
        if len(current_block_lines) > 0:
            last_line = current_block_lines[-1]
            if (
                line.rect.y0 > last_line.rect.y1 + 5
                or (  # upper boundary of line is higher up than lower boundary plus 5 points of last line.
                    threshold and line.rect.y0 > last_line.rect.y1 and line.rect.y0 > last_line.rect.y0 + threshold
                )
            ):
                blocks.append(TextBlock(current_block_lines))
                current_block_lines = []

        current_block_lines.append(line)
    if len(current_block_lines):
        blocks.append(TextBlock(current_block_lines))

    return blocks


def _block_separated_by_line(
    last_line: TextLine, current_line: TextLine, block: TextBlock, geometric_lines: list[Line], threshold: float
) -> bool:
    """Check if a block is separated by a line.

    Args:
        last_line (TextLine): The previous line.
        current_line (TextLine): The current line.
        block (TextBlock): Current block.
        geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        threshold (float): Percentage of the block width that needs to be covered by a line.

    Returns:
        bool: True if the block is separated by a line, False otherwise.
    """
    last_line_y_coordinate = (last_line.rect.y0 + last_line.rect.y1) / 2
    current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2
    for line in geometric_lines:
        line_left_x = np.min([line.start.x, line.end.x])
        line_right_x = np.max([line.start.x, line.end.x])
        line_y_coordinate = (line.start.y + line.end.y) / 2

        is_line_long_enough = (
            np.min([block.rect.x1, line_right_x]) - np.max([block.rect.x0, line_left_x]) > threshold * block.rect.width
        )

        line_ends_block = last_line_y_coordinate < line_y_coordinate and line_y_coordinate < current_line_y_coordinate
        if is_line_long_enough and line_ends_block:
            return True
    return False


def _block_separated_by_lefthandside_line_segment(
    last_line: TextLine, current_line: TextLine, block: TextBlock, geometric_lines: list[Line], length_threshold: int
) -> bool:
    """Check if a block is separated by a line segment on the left side of the block.

    Args:
        last_line (TextLine): The previous line.
        current_line (TextLine): The current line.
        block (TextBlock): Current block.
        geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        length_threshold (int): The minimum length of a line segment on the left side of a block to split it.

    Returns:
        bool: True if the block is separated by a line segment, False otherwise.
    """
    last_line_y_coordinate = (last_line.rect.y0 + last_line.rect.y1) / 2
    current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2

    for line in geometric_lines:
        line_y_coordinate = (line.start.y + line.end.y) / 2

        line_cuts_lefthandside_of_block = line.start.x < block.rect.x0 and line.end.x > block.rect.x0
        is_line_long_enough = (
            np.abs(line.start.x - line.end.x) > length_threshold
        )  # for the block splitting, we only care about x-extension

        line_ends_block = last_line_y_coordinate < line_y_coordinate and line_y_coordinate < current_line_y_coordinate

        weak_condition = (
            (line.start.x - 5 < block.rect.x0 and line.end.x > block.rect.x0)
            and (np.abs(line.start.x - line.end.x) > length_threshold - 2)
            and (len(block.lines) > 1)
        )  # if block has at least three lines, we weaken the splitting condition.
        # It is three lines because the statement means that block.lines has at least two elements.
        # The third line is current_line

        if line_ends_block and ((is_line_long_enough and line_cuts_lefthandside_of_block) or weak_condition):
            return True
    return False
