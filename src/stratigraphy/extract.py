"""Contains the main extraction pipeline for stratigraphy."""

from dataclasses import dataclass

import fitz
import rtree

from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.depth import AAboveBInterval, Interval
from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBox, BoundingBoxes
from stratigraphy.depths_materials_column_pairs.material_description_rect_with_sidebar import (
    MaterialDescriptionRectWithSidebar,
)
from stratigraphy.layer.layer import IntervalBlockPair, Layer
from stratigraphy.lines.line import TextLine
from stratigraphy.sidebar import (
    AAboveBSidebarExtractor,
    AToBSidebarExtractor,
    LayerIdentifierSidebarExtractor,
    Sidebar,
)
from stratigraphy.sidebar.sidebar import SidebarNoise, noise_count
from stratigraphy.text.find_description import (
    get_description_blocks,
    get_description_lines,
)
from stratigraphy.text.textblock import MaterialDescription, MaterialDescriptionLine, TextBlock, block_distance
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.util import (
    x_overlap,
    x_overlap_significant_smallest,
)


@dataclass
class ProcessPageResult:
    """The result of processing a single page of a pdf."""

    predictions: list[Layer]
    bounding_boxes: list[BoundingBoxes]


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
    # Detect Layer Identifier Sidebars

    layer_identifier_sidebars = LayerIdentifierSidebarExtractor.from_lines(lines)
    material_descriptions_sidebar_pairs = []
    for layer_identifier_sidebar in layer_identifier_sidebars:
        material_description_rect = find_material_description_column(
            lines, layer_identifier_sidebar, language, **params["material_description"]
        )
        if material_description_rect:
            material_descriptions_sidebar_pairs.append(
                MaterialDescriptionRectWithSidebar(layer_identifier_sidebar, material_description_rect)
            )

    if material_descriptions_sidebar_pairs:
        material_descriptions_sidebar_pairs.sort(key=lambda pair: pair.score_match)

    # If there is a layer identifier sidebar, then we use this directly.
    # Else, we search for sidebars with depths.
    # We could also think of some scoring mechanism to decide which one to use.
    if not material_descriptions_sidebar_pairs:
        words = sorted([word for line in lines for word in line.words], key=lambda word: word.rect.y0)
        word_rtree = rtree.index.Index()
        for word in words:
            word_rtree.insert(id(word), (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1), obj=word)

        a_to_b_sidebars = AToBSidebarExtractor.find_in_words(words)
        used_entry_rects = []
        for column in a_to_b_sidebars:
            for entry in column.entries:
                used_entry_rects.extend([entry.start.rect, entry.end.rect])

        # create sidebars with noise count
        sidebars_noise: list[SidebarNoise] = [
            SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, word_rtree)) for sidebar in a_to_b_sidebars
        ]
        sidebars_noise.extend(
            AAboveBSidebarExtractor.find_in_words(
                words, word_rtree, used_entry_rects, sidebar_params=params["depth_column_params"]
            )
        )

        for sidebar_noise in sidebars_noise:
            material_description_rect = find_material_description_column(
                lines, sidebar_noise.sidebar, language, **params["material_description"]
            )
            if material_description_rect:
                material_descriptions_sidebar_pairs.append(
                    MaterialDescriptionRectWithSidebar(
                        sidebar=sidebar_noise.sidebar,
                        material_description_rect=material_description_rect,
                        noise_count=sidebar_noise.noise_count,
                    )
                )
        # lowest score first
        material_descriptions_sidebar_pairs.sort(key=lambda pair: pair.score_match)

    to_delete = []
    for i, pair in enumerate(material_descriptions_sidebar_pairs):
        if any(
            pair.material_description_rect.intersects(other_pair.material_description_rect)
            for other_pair in material_descriptions_sidebar_pairs[i + 1 :]
        ):
            to_delete.append(i)
    filtered_pairs = [item for index, item in enumerate(material_descriptions_sidebar_pairs) if index not in to_delete]

    pairs: list[IntervalBlockPair] = []  # list of matched depth intervals and text blocks
    if filtered_pairs:  # match sidebars with material description
        bounding_boxes = [
            BoundingBoxes.from_material_description_rect_with_sidebar(pair, page_number) for pair in filtered_pairs
        ]
        for pair in filtered_pairs:
            description_lines = get_description_lines(lines, pair.material_description_rect)
            if len(description_lines) > 1:
                new_pairs = match_columns(
                    pair.sidebar, description_lines, geometric_lines, pair.material_description_rect, **params
                )
                pairs.extend(new_pairs)
    else:
        # Fallback when no depth column was found
        material_description_rect = find_material_description_column(
            lines, sidebar=None, language=language, **params["material_description"]
        )
        bounding_boxes = []
        if material_description_rect:
            bounding_boxes.append(
                BoundingBoxes(
                    sidebar_bbox=None,
                    depth_column_entry_bboxes=[],
                    material_description_bbox=BoundingBox(material_description_rect),
                    page=page_number,
                )
            )
            description_lines = get_description_lines(lines, material_description_rect)
            description_blocks = get_description_blocks(
                description_lines,
                geometric_lines,
                material_description_rect,
                params["block_line_ratio"],
                params["left_line_length_threshold"],
            )
            pairs.extend([IntervalBlockPair(block=block, depth_interval=None) for block in description_blocks])

    layer_predictions = [
        Layer(
            material_description=FeatureOnPage(
                feature=MaterialDescription(
                    text=pair.block.text,
                    lines=[
                        FeatureOnPage(
                            feature=MaterialDescriptionLine(text_line.text),
                            rect=text_line.rect,
                            page=text_line.page_number,
                        )
                        for text_line in pair.block.lines
                    ],
                ),
                rect=pair.block.rect,
                page=page_number,
            ),
            # TODO LGVISIUM-104 don't automatically convert any interval to an AAboveBInterval
            depth_interval=AAboveBInterval(start=pair.depth_interval.start, end=pair.depth_interval.end)
            if pair.depth_interval
            else None,
        )
        for pair in pairs
    ]
    layer_predictions = [layer for layer in layer_predictions if layer.description_nonempty()]
    return ProcessPageResult(layer_predictions, bounding_boxes)


def match_columns(
    sidebar: Sidebar,
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    material_description_rect: fitz.Rect,
    **params: dict,
) -> list[IntervalBlockPair]:
    """Match the layers that can be derived from the sidebar with the description lines.

    This function identifies groups of depth intervals and text blocks that are likely to match.
    The actual matching between text blocks and depth intervals is handled by the implementation of the actual Sidebar
    instance (e.b. AAboveBSidebar, AToBSidebar).

    Args:
        sidebar (Sidebar): The sidebar.
        description_lines (list[TextLine]): The description lines.
        geometric_lines (list[Line]): The geometric lines.
        material_description_rect (fitz.Rect): The material description rectangle.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[IntervalBlockPair]: The matched depth intervals and text blocks.
    """
    return [
        element
        for group in sidebar.identify_groups(description_lines, geometric_lines, material_description_rect, **params)
        for element in transform_groups(group.depth_intervals, group.blocks, **params)
    ]


def transform_groups(
    depth_intervals: list[Interval], blocks: list[TextBlock], **params: dict
) -> list[IntervalBlockPair]:
    """Transforms the text blocks such that their number equals the number of depth intervals.

    If there are more depth intervals than text blocks, text blocks are splitted. When there
    are more text blocks than depth intervals, text blocks are merged. If the number of text blocks
    and depth intervals equals, we proceed with the pairing.

    Args:
        depth_intervals (List[Interval]): The depth intervals from the pdf.
        blocks (List[TextBlock]): Found textblocks from the pdf.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        List[IntervalBlockPair]: Pairing of text blocks and depth intervals.
    """
    if len(depth_intervals) <= 1:
        concatenated_block = TextBlock(
            [line for block in blocks for line in block.lines]
        )  # concatenate all text lines within a block; line separation flag does not matter here.
        depth_interval = depth_intervals[0] if len(depth_intervals) else None
        return [IntervalBlockPair(depth_interval=depth_interval, block=concatenated_block)]
    else:
        if len(blocks) < len(depth_intervals):
            blocks = split_blocks_by_textline_length(blocks, target_split_count=len(depth_intervals) - len(blocks))

        if len(blocks) > len(depth_intervals):
            # create additional depth intervals with end & start value None to match the number of blocks
            depth_intervals.extend([AAboveBInterval(None, None) for _ in range(len(blocks) - len(depth_intervals))])

        return [
            IntervalBlockPair(depth_interval=depth_interval, block=block)
            for depth_interval, block in zip(depth_intervals, blocks, strict=False)
        ]


def merge_blocks_by_vertical_spacing(blocks: list[TextBlock], target_merge_count: int) -> list[TextBlock]:
    """Merge textblocks without any geometric lines that separates them.

    Note: Deprecated. Currently not in use anymore. Kept here until we are sure that it is not needed anymore.

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
    lines: list[TextLine], sidebar: Sidebar | None, language: str, **params: dict
) -> fitz.Rect | None:
    """Find the material description column given a depth column.

    Args:
        lines (list[TextLine]): The text lines of the page.
        sidebar (Sidebar | None): The sidebar to be associated with the material descriptions.
        language (str): The language of the page.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        fitz.Rect | None: The material description column.
    """
    if sidebar:
        above_sidebar = [
            line for line in lines if x_overlap(line.rect, sidebar.rect()) and line.rect.y0 < sidebar.rect().y0
        ]

        min_y0 = max(line.rect.y0 for line in above_sidebar) if above_sidebar else -1

        def check_y0_condition(y0):
            return y0 > min_y0 and y0 < sidebar.rect().y1
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
    if sidebar:
        return max(
            candidate_rects,
            key=lambda rect: MaterialDescriptionRectWithSidebar(sidebar, rect).score_match,
        )
    else:
        return candidate_rects[0]
