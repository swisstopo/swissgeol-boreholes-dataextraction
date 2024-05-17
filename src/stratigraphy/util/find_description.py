"""This module contains functions to find the description (blocks) of a material in a pdf page."""

import fitz

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.description_block_splitter import (
    SplitDescriptionBlockByLeftHandSideSeparator,
    SplitDescriptionBlockByLine,
    SplitDescriptionBlockByVerticalSpace,
)
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
    if not lines:
        return []
    filtered_lines = [
        line
        for line in lines
        if line.rect.x0 < material_description_rect.x1 - 0.4 * material_description_rect.width
        if material_description_rect.contains(line.rect)
    ]
    return sorted([line for line in filtered_lines if line], key=lambda line: line.rect.y0)


def get_description_blocks_from_layer_index(
    layer_index_entries: list[TextLine], description_lines: list[TextLine]
) -> list[TextBlock]:
    """Divide the description lines into blocks based on the layer index entries.

    Args:
        layer_index_entries (list[TextLine]): The layer index entries.
        description_lines (list[TextLine]): All lines constituting the material description.

    Returns:
        list[TextBlock]: The blocks of the material description.
    """
    blocks = []
    line_index = 0
    for layer_index_idx, _layer_index in enumerate(layer_index_entries):
        # don't allow a layer above depth 0

        next_layer_index = (
            layer_index_entries[layer_index_idx + 1] if layer_index_idx + 1 < len(layer_index_entries) else None
        )

        matched_block = matching_blocks(description_lines, line_index, next_layer_index)
        line_index += sum([len(block.lines) for block in matched_block])
        blocks.extend(matched_block)

    return blocks


def matching_blocks(all_lines: list[TextLine], line_index: int, next_layer_index: TextLine | None) -> list[TextBlock]:
    """Adds lines to a block until the next layer index is reached.

    Args:
        all_lines (list[TextLine]): All TextLine objects constituting the material description.
        line_index (int): The index of the last line that is already assigned to a block.
        next_layer_index (TextLine | None): The next layer index.

    Returns:
        list[TextBlock]: The next block or an empty list if no lines are added.
    """
    y1_threshold = None
    if next_layer_index:
        next_interval_start_rect = next_layer_index.rect
        y1_threshold = next_interval_start_rect.y0 + next_interval_start_rect.height / 2

    matched_lines = []

    for current_line in all_lines[line_index:]:
        if y1_threshold is None or current_line.rect.y1 < y1_threshold:
            matched_lines.append(current_line)
        else:
            break

    if len(matched_lines):
        return [TextBlock(matched_lines)]
    else:
        return []


def get_description_blocks(
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    material_description_rect: fitz.Rect,
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
        material_description_rect (fitz.Rect): The bounding box of the material descriptions.
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
