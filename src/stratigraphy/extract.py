"""Contains the main extraction pipeline for stratigraphy."""

import logging
import math
from dataclasses import dataclass

import fitz

from stratigraphy.depthcolumn import find_depth_columns
from stratigraphy.depthcolumn.depthcolumn import DepthColumn
from stratigraphy.depths_materials_column_pairs.depths_materials_column_pairs import DepthsMaterialsColumnPairs
from stratigraphy.layer.layer import IntervalBlockGroup, Layer, LayersOnPage
from stratigraphy.layer.layer_identifier_column import (
    LayerIdentifierColumn,
    find_layer_identifier_column,
    find_layer_identifier_column_entries,
)
from stratigraphy.lines.line import TextLine, TextWord
from stratigraphy.text.find_description import (
    get_description_blocks,
    get_description_blocks_from_layer_identifier,
    get_description_lines,
)
from stratigraphy.text.textblock import TextBlock, block_distance
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.interval import BoundaryInterval, Interval
from stratigraphy.util.util import (
    x_overlap,
    x_overlap_significant_smallest,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessPageResult:
    """The result of processing a single page of a pdf."""

    predictions: LayersOnPage
    depth_material_pairs: list[DepthsMaterialsColumnPairs]


def process_page(
    lines: list[TextLine], geometric_lines, language: str, page_number: int, **params: dict
) -> ProcessPageResult:
    """Process a single page of a pdf.

    # TODO: Ideally, one function does one thing. This function does a lot of things. It should be split into
    # smaller functions.

    Finds all descriptions and depth intervals on the page and matches them.

    Args:
        lines (list[TextLine]): all the text lines on the page.
        geometric_lines (list[Line]): The geometric lines of the page.
        language (str): The language of the page.
        page_number (int): The page number.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[dict]: All list of the text of all description blocks.
    """
    # Detect Layer Index Columns
    layer_identifier_entries = find_layer_identifier_column_entries(lines)
    layer_identifier_columns = (
        find_layer_identifier_column(layer_identifier_entries) if layer_identifier_entries else []
    )
    pairs = []
    if layer_identifier_columns:
        for layer_identifier_column in layer_identifier_columns:
            material_description_rect = find_material_description_column(
                lines, layer_identifier_column, language, **params["material_description"]
            )
            if material_description_rect:
                pairs.append((layer_identifier_column, material_description_rect))

        # Obtain the best pair. In contrast to depth columns, there only ever is one layer index column per page.
        if pairs:
            pairs.sort(key=lambda pair: score_column_match(pair[0], pair[1]))

    words = [word for line in lines for word in line.words]

    # If there is a layer identifier column, then we use this directly.
    # Else, we search for depth columns. We could also think of some scoring mechanism to decide which one to use.
    if not pairs:
        depth_column_entries = find_depth_columns.depth_column_entries(words, include_splits=True)
        layer_depth_columns = find_depth_columns.find_layer_depth_columns(depth_column_entries, words)

        used_entry_rects = []
        for column in layer_depth_columns:
            for entry in column.entries:
                used_entry_rects.extend([entry.start.rect, entry.end.rect])

        depth_column_entries = [
            entry
            for entry in find_depth_columns.depth_column_entries(words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]
        depth_columns: list[DepthColumn] = layer_depth_columns
        depth_columns.extend(
            find_depth_columns.find_depth_columns(
                depth_column_entries, words, page_number, depth_column_params=params["depth_column_params"]
            )
        )

        for depth_column in depth_columns:
            material_description_rect = find_material_description_column(
                lines, depth_column, language, **params["material_description"]
            )
            if material_description_rect:
                pairs.append((depth_column, material_description_rect))
        # lowest score first
        pairs.sort(key=lambda pair: score_column_match(pair[0], pair[1], words))

    to_delete = []
    for i, (_depth_column, material_description_rect) in enumerate(pairs):
        if any(material_description_rect.intersects(other_rect) for _, other_rect in pairs[i + 1 :]):
            to_delete.append(i)
    filtered_pairs = [item for index, item in enumerate(pairs) if index not in to_delete]

    groups: list[IntervalBlockGroup] = []  # list of matched depth intervals and text blocks
    # groups is of the form: [{"depth_interval": BoundaryInterval, "block": TextBlock}]
    if filtered_pairs:  # match depth column items with material description
        for depth_column, material_description_rect in filtered_pairs:
            description_lines = get_description_lines(lines, material_description_rect)
            if len(description_lines) > 1:
                new_groups = match_columns(
                    depth_column, description_lines, geometric_lines, material_description_rect, **params
                )
                groups.extend(new_groups)
        filtered_depth_material_column_pairs = [
            DepthsMaterialsColumnPairs(
                depth_column=depth_column, material_description_rect=material_description_rect, page=page_number
            )
            for depth_column, material_description_rect in filtered_pairs
        ]
    else:
        filtered_depth_material_column_pairs = []
        # Fallback when no depth column was found
        material_description_rect = find_material_description_column(
            lines, depth_column=None, language=language, **params["material_description"]
        )
        if material_description_rect:
            description_lines = get_description_lines(lines, material_description_rect)
            description_blocks = get_description_blocks(
                description_lines,
                geometric_lines,
                material_description_rect,
                params["block_line_ratio"],
                params["left_line_length_threshold"],
            )
            groups.extend([IntervalBlockGroup(block=block, depth_interval=None) for block in description_blocks])
            filtered_depth_material_column_pairs.extend(
                [
                    DepthsMaterialsColumnPairs(
                        depth_column=None, material_description_rect=material_description_rect, page=page_number
                    )
                ]
            )

    layer_predictions = LayersOnPage(
        [
            Layer(
                material_description=group.block,
                depth_interval=BoundaryInterval(start=group.depth_interval.start, end=group.depth_interval.end)
                if group.depth_interval
                else None,
            )
            for group in groups
        ]
    )
    layer_predictions.remove_empty_predictions()
    return ProcessPageResult(layer_predictions, filtered_depth_material_column_pairs)


def score_column_match(
    depth_column: DepthColumn, material_description_rect: fitz.Rect, all_words: list[TextWord] | None = None
) -> float:
    """Scores the match between a depth column and a material description.

    Args:
        depth_column (DepthColumn): The depth column.
        material_description_rect (fitz.Rect): The material description rectangle.
        all_words (list[TextWord] | None, optional): List of the available text words. Defaults to None.

    Returns:
        float: The score of the match.
    """
    rect = depth_column.rect()
    top = rect.y0
    bottom = rect.y1
    right = rect.x1
    distance = (
        abs(top - material_description_rect.y0)
        + abs(bottom - material_description_rect.y1)
        + abs(right - material_description_rect.x0)
    )

    height = bottom - top

    noise_count = depth_column.noise_count(all_words) if all_words else 0

    return (height - distance) * math.pow(0.8, noise_count)


def match_columns(
    depth_column: DepthColumn | LayerIdentifierColumn,
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    material_description_rect: fitz.Rect,
    **params: dict,
) -> list[IntervalBlockGroup]:
    """Match the depth column entries with the description lines.

    This function identifies groups of depth intervals and text blocks that are likely to match.
    Makes a distinction between DepthColumn and LayerIdentifierColumn and obtains the corresponding text blocks
    as well as their depth intervals where present.

    Args:
        depth_column (DepthColumn | LayerIdentifierColumn): The depth column.
        description_lines (list[TextLine]): The description lines.
        geometric_lines (list[Line]): The geometric lines.
        material_description_rect (fitz.Rect): The material description rectangle.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[IntervalBlockGroup]: The matched depth intervals and text blocks.
    """
    if isinstance(depth_column, DepthColumn):
        return [
            element
            for group in depth_column.identify_groups(
                description_lines, geometric_lines, material_description_rect, **params
            )
            for element in transform_groups(group.depth_interval, group.block, **params)
        ]
    elif isinstance(depth_column, LayerIdentifierColumn):
        blocks = get_description_blocks_from_layer_identifier(depth_column.entries, description_lines)
        groups: list[IntervalBlockGroup] = []
        for block in blocks:
            depth_interval = find_depth_columns.get_depth_interval_from_textblock(block)
            if depth_interval:
                groups.append(IntervalBlockGroup(depth_interval=depth_interval, block=block))
            else:
                groups.append(IntervalBlockGroup(depth_interval=None, block=block))
        return groups
    else:
        raise ValueError(
            f"depth_column must be a DepthColumn or a LayerIdentifierColumn object. Got {type(depth_column)}."
        )


def transform_groups(
    depth_intervals: list[Interval], blocks: list[TextBlock], **params: dict
) -> list[IntervalBlockGroup]:
    """Transforms the text blocks such that their number equals the number of depth intervals.

    If there are more depth intervals than text blocks, text blocks are splitted. When there
    are more text blocks than depth intervals, text blocks are merged. If the number of text blocks
    and depth intervals equals, we proceed with the pairing.

    Args:
        depth_intervals (List[Interval]): The depth intervals from the pdf.
        blocks (List[TextBlock]): Found textblocks from the pdf.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        List[IntervalBlockGroup]: Pairing of text blocks and depth intervals.
    """
    if len(depth_intervals) == 0:
        return []
    elif len(depth_intervals) == 1:
        concatenated_block = TextBlock(
            [line for block in blocks for line in block.lines]
        )  # concatenate all text lines within a block; line separation flag does not matter here.
        return [IntervalBlockGroup(depth_interval=depth_intervals[0], block=concatenated_block)]
    else:
        if len(blocks) < len(depth_intervals):
            blocks = split_blocks_by_textline_length(blocks, target_split_count=len(depth_intervals) - len(blocks))

        if len(blocks) > len(depth_intervals):
            # create additional depth intervals with end & start value None to match the number of blocks
            depth_intervals.extend([BoundaryInterval(None, None) for _ in range(len(blocks) - len(depth_intervals))])

        return [
            IntervalBlockGroup(depth_interval=depth_interval, block=block)
            for depth_interval, block in zip(depth_intervals, blocks, strict=False)
        ]


def merge_blocks_by_vertical_spacing(blocks: list[TextBlock], target_merge_count: int) -> list[TextBlock]:
    """Merge textblocks without any geometric lines that separates them.

    Note: Deprecated. Currently not in use any more. Kept here until we are sure that it is not needed anymore.

    The logic looks at the distances between the textblocks and merges them if they are closer
    than a certain cutoff.

    Args:
        blocks (List[TextBlock]): Textblocks that are to be merged.
        target_merge_count (int): the number of merges that we'd like to happen (i.e. we'd like the total number of
                                  blocks to be reduced by this number)

    Returns:
        List[TextBlock]: The merged textblocks.
    """
    distances = []
    for block_index in range(len(blocks) - 1):
        distances.append(block_distance(blocks[block_index], blocks[block_index + 1]))
    cutoff = sorted(distances)[target_merge_count - 1]  # merge all blocks that have a distance smaller than this
    merged_count = 0
    merged_blocks = []
    current_merged_block = blocks[0]
    for block_index in range(len(blocks) - 1):
        new_block = blocks[block_index + 1]
        if (
            merged_count < target_merge_count
            and block_distance(blocks[block_index], blocks[block_index + 1]) <= cutoff
        ):
            current_merged_block = current_merged_block.concatenate(new_block)
            merged_count += 1
        else:
            merged_blocks.append(current_merged_block)
            current_merged_block = new_block

    if current_merged_block.lines:
        merged_blocks.append(current_merged_block)
    return merged_blocks


def split_blocks_by_textline_length(blocks: list[TextBlock], target_split_count: int) -> list[TextBlock]:
    """Split textblocks without any geometric lines that separates them.

    The logic looks at the lengths of the text lines and cuts them off
    if there are textlines that are shorter than others.
    # TODO: Extend documentation about logic.

    Args:
        blocks (List[TextBlock]): Textblocks that are to be split.
        target_split_count (int): the number of splits that we'd like to happen (i.e. we'd like the total number of
                                  blocks to be increased by this number)

    Returns:
        List[TextBlock]: The split textblocks.
    """
    line_lengths = sorted([line.rect.x1 for block in blocks for line in block.lines[:-1]])
    if len(line_lengths) <= target_split_count:  # In that case each line is a block
        return [TextBlock([line]) for block in blocks for line in block.lines]
    else:
        cutoff_values = line_lengths[:target_split_count]  # all lines inside cutoff_values will be split line
        split_blocks = []
        current_block_lines = []
        for block in blocks:
            for line_index in range(block.line_count):
                line = block.lines[line_index]
                current_block_lines.append(line)
                if line_index < block.line_count - 1 and line.rect.x1 in cutoff_values:
                    split_blocks.append(TextBlock(current_block_lines))
                    cutoff_values.remove(line.rect.x1)
                    current_block_lines = []
            if current_block_lines:
                split_blocks.append(TextBlock(current_block_lines))
                current_block_lines = []
            if (
                block.is_terminated_by_line
            ):  # If block was terminated by a line, populate the flag to the last element of split_blocks.
                split_blocks[-1].is_terminated_by_line = True
        return split_blocks


def find_material_description_column(
    lines: list[TextLine], depth_column: DepthColumn | None, language: str, **params: dict
) -> fitz.Rect | None:
    """Find the material description column given a depth column.

    Args:
        lines (list[TextLine]): The text lines of the page.
        depth_column (DepthColumn | None): The depth column.
        language (str): The language of the page.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        fitz.Rect | None: The material description column.
    """
    if depth_column:
        above_depth_column = [
            line
            for line in lines
            if x_overlap(line.rect, depth_column.rect()) and line.rect.y0 < depth_column.rect().y0
        ]

        min_y0 = max(line.rect.y0 for line in above_depth_column) if above_depth_column else -1

        def check_y0_condition(y0):
            return y0 > min_y0 and y0 < depth_column.rect().y1
    else:

        def check_y0_condition(y0):
            return True

    candidate_description = [line for line in lines if check_y0_condition(line.rect.y0)]
    is_description = [line for line in candidate_description if line.is_description(params[language])]

    if len(candidate_description) == 0:
        return

    description_clusters = []
    while len(is_description) > 0:
        coverage_by_generating_line = [
            [other for other in is_description if x_overlap_significant_smallest(line.rect, other.rect, 0.5)]
            for line in is_description
        ]

        def filter_coverage(coverage):
            if coverage:
                min_x0 = min(line.rect.x0 for line in coverage)
                max_x1 = max(line.rect.x1 for line in coverage)
                x0_threshold = max_x1 - 0.4 * (
                    max_x1 - min_x0
                )  #  how did we determine the 0.4? Should it be a parameter? What would it do if we were to change it?
                return [line for line in coverage if line.rect.x0 < x0_threshold]
            else:
                return []

        coverage_by_generating_line = [filter_coverage(coverage) for coverage in coverage_by_generating_line]
        max_coverage = max(coverage_by_generating_line, key=len)
        description_clusters.append(max_coverage)
        is_description = [line for line in is_description if line not in max_coverage]

    candidate_rects = []

    for cluster in description_clusters:
        best_y0 = min([line.rect.y0 for line in cluster])
        best_y1 = max([line.rect.y1 for line in cluster])

        min_description_x0 = min(
            [
                line.rect.x0 - 0.01 * line.rect.width for line in cluster
            ]  # How did we determine the 0.01? Should it be a parameter? What would it do if we were to change it?
        )
        max_description_x0 = max(
            [
                line.rect.x0 + 0.2 * line.rect.width for line in cluster
            ]  # How did we determine the 0.2? Should it be a parameter? What would it do if we were to change it?
        )
        good_lines = [
            line
            for line in candidate_description
            if line.rect.y0 >= best_y0 and line.rect.y1 <= best_y1
            if min_description_x0 < line.rect.x0 < max_description_x0
        ]
        best_x0 = min([line.rect.x0 for line in good_lines])
        best_x1 = max([line.rect.x1 for line in good_lines])

        # expand to include entire last block
        def is_below(best_x0, best_y1, line):
            return (
                (
                    line.rect.x0 > best_x0 - 5
                )  # How did we determine the 5? Should it be a parameter? What would it do if we were to change it?
                and (line.rect.x0 < (best_x0 + best_x1) / 2)  # noqa B023
                and (
                    line.rect.y0 < best_y1 + 10
                )  # How did we determine the 10? Should it be a parameter? What would it do if we were to change it?
                and (line.rect.y1 > best_y1)
            )

        continue_search = True
        while continue_search:
            line = next((line for line in lines if is_below(best_x0, best_y1, line)), None)
            if line:
                best_x0 = min(best_x0, line.rect.x0)
                best_x1 = max(best_x1, line.rect.x1)
                best_y1 = line.rect.y1
            else:
                continue_search = False

        candidate_rects.append(fitz.Rect(best_x0, best_y0, best_x1, best_y1))

    if len(candidate_rects) == 0:
        return None
    if depth_column:
        return max(candidate_rects, key=lambda rect: score_column_match(depth_column, rect))
    else:
        return candidate_rects[0]
