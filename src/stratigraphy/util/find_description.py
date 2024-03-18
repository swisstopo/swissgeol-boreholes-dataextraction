import fitz
import numpy as np

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.line import TextLine
from stratigraphy.util.textblock import TextBlock


def get_description_lines(lines: list[TextLine], material_description_rect: fitz.Rect) -> list[TextLine]:
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
    target_layer_count: int = None,
) -> list[TextBlock]:
    distances = []
    for line_index in range(len(description_lines) - 1):
        line1rect = description_lines[line_index].rect
        line2rect = description_lines[line_index + 1].rect
        if line2rect.y0 > line1rect.y0 + line1rect.height / 2:
            distances.append(line2rect.y0 - line1rect.y0)

    threshold = None
    if len(distances):
        threshold = min(distances) * 1.15

    blocks = []
    current_block_lines = []
    for line in description_lines:
        if len(current_block_lines) > 0:
            last_line = current_block_lines[-1]
            if _block_separated_by_line(
                last_line, line, TextBlock(current_block_lines), geometric_lines, threshold=block_line_ratio
            ):
                blocks.append(TextBlock(current_block_lines, is_terminated_by_line=True))
                current_block_lines = []

        current_block_lines.append(line)
    if len(current_block_lines):
        blocks.append(TextBlock(current_block_lines))

    if target_layer_count is None:
        # If we have only found one splitting line, then we fall back to considering vertical spacing, as it is more
        # likely that this line is a false positive, than that we have a borehole profile with only two layers.
        min_block_count = 3
    else:
        # If the number of blocks is less than 2/3 of the expected number of layers (based on the information from the
        # depth column, then the splitting based on horizontal lines is not reliable, and we fall back to considering
        # vertical spacing between text.
        min_block_count = 2 / 3 * target_layer_count

    if len(blocks) < min_block_count:
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

    blocks = [new_block for block in blocks for new_block in block.split_based_on_indentation()]

    return blocks


def _block_separated_by_line(
    last_line: TextLine, current_line: TextLine, block: TextBlock, geometric_lines: list[Line], threshold
) -> bool:
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
