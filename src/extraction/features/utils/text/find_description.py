"""This module contains functions to find the description (blocks) of a material in a pdf page."""

import pymupdf
from extraction.features.utils.geometry.geometry_dataclasses import Line

from .description_block_splitter import (
    SplitDescriptionBlockByLeftHandSideSeparator,
    SplitDescriptionBlockByLine,
    SplitDescriptionBlockByVerticalSpace,
)
from .textblock import TextBlock
from .textline import TextLine


def get_description_lines(lines: list[TextLine], material_description_rect: pymupdf.Rect) -> list[TextLine]:
    """Get the description lines of a material.

    Checks if the lines are within the material description rectangle and if they are not too far to the right.

    Args:
        lines (list[TextLine]): The lines to filter.
        material_description_rect (pymupdf.Rect): The rectangle containing the material description.

    Returns:
        list[TextLine]: The filtered lines.
    """
    if not lines:
        return []
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
    material_description_rect: pymupdf.Rect,
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
        material_description_rect (pymupdf.Rect): The bounding box of the material descriptions.
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
    if distances:
        threshold = min(distances) * 1.15

    # Create blocks separated by lines
    blocks = SplitDescriptionBlockByLine(
        threshold=block_line_ratio,
        material_description_rect=material_description_rect,
        geometric_lines=geometric_lines,
    ).create_blocks(description_lines)

    # Create blocks separated by lefthandside line segments
    _blocks = []
    splitter = SplitDescriptionBlockByLeftHandSideSeparator(
        length_threshold=left_line_length_threshold, geometric_lines=geometric_lines
    )
    for block in blocks:
        _blocks.extend(splitter.create_blocks(block.lines))
        if block.is_terminated_by_line:  # keep the line termination if it was there
            _blocks[-1].is_terminated_by_line = True
    blocks = _blocks

    min_block_count = 3 if target_layer_count is None else 2 / 3 * target_layer_count
    # If we have only found one splitting line, then we fall back to considering vertical spacing, as it is more
    # likely that this line is a false positive, than that we have a borehole profile with only two layers.
    # If the number of blocks is less than 2/3 of the expected number of layers (based on the information from the
    # depth column, then the splitting based on horizontal lines is not reliable, and we fall back to considering
    # vertical spacing between text.

    splitter = SplitDescriptionBlockByVerticalSpace(threshold=threshold)

    count_blocks_divided_by_line = len([block for block in blocks if block.is_terminated_by_line])
    if len(blocks) < min_block_count:
        # This case means that there are fewer blocks than the minimum number of blocks we expect.
        # In this case we redo all the blocks from scratch.
        blocks = splitter.create_blocks(description_lines)

    elif count_blocks_divided_by_line < min_block_count:
        # In this case the blocks are due to line segments. However, they are mostly due to small segments
        # on the lefthandside of the blocks. Minimum there are fewer blocks due to lines than min_block_count.
        # Often, these lefthandside lines are only used when space is tight. If space is not tight, those
        # indicators are dropped. That's why we have to consider vertical spacing as well.
        _blocks = []
        for block in blocks:
            _blocks.extend(splitter.create_blocks(block.lines))
        blocks = _blocks
    blocks = [new_block for block in blocks for new_block in block.split_based_on_indentation()]
    blocks = [block for block in blocks if not block._is_legend()]
    return blocks
