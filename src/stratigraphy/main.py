"""This module contains the main pipeline for the boreholes data extraction."""

import json
import logging
import math
import os
from pathlib import Path

import fitz
from dotenv import load_dotenv

from stratigraphy import DATAPATH
from stratigraphy.benchmark.score import evaluate_matching
from stratigraphy.line_detection import extract_lines, line_detection_params
from stratigraphy.util import find_depth_columns
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.depthcolumn import DepthColumn
from stratigraphy.util.find_description import get_description_blocks, get_description_lines
from stratigraphy.util.interval import Interval
from stratigraphy.util.line import DepthInterval, TextLine
from stratigraphy.util.textblock import TextBlock, block_distance
from stratigraphy.util.util import (
    flatten,
    parse_and_remove_empty_predictions,
    read_params,
    x_overlap,
    x_overlap_significant_smallest,
)

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matching_params = read_params("matching_params.yml")


def process_page(page: fitz.Page, **params: dict) -> list[dict]:
    """Process a single page of a pdf.

    Finds all descriptions and depth intervals on the page and matches them.

    Args:
        page (fitz.Page): The page to process.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[dict]: All list of the text of all description blocks.
    """
    words = []
    words_by_line = {}
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in fitz.utils.get_text(page, "words"):
        rect = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
        depth_interval = DepthInterval(rect, word)
        words.append(TextLine([depth_interval]))
        key = f"{block_no}_{line_no}"
        if key not in words_by_line:
            words_by_line[key] = []
        words_by_line[key].append(depth_interval)

    raw_lines = [TextLine(words_by_line[key]) for key in words_by_line]

    lines = []
    current_line_words = []
    for line_index, raw_line in enumerate(raw_lines):
        for word_index, word in enumerate(raw_line.words):
            remaining_line = TextLine(raw_line.words[word_index:])
            if len(current_line_words) > 0 and remaining_line.is_line_start(lines, raw_lines[line_index + 1 :]):
                lines.append(TextLine(current_line_words))
                current_line_words = []
            current_line_words.append(word)
        if len(current_line_words):
            lines.append(TextLine(current_line_words))
            current_line_words = []

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
    depth_columns.extend(find_depth_columns.find_depth_columns(depth_column_entries, words))

    pairs = []
    for depth_column in depth_columns:
        material_description_rect = find_material_description_column(lines, depth_column)
        if material_description_rect:
            pairs.append((depth_column, material_description_rect))

    # lowest score first
    pairs.sort(key=lambda pair: score_column_match(pair[0], pair[1], words))

    to_delete = []
    for i, (_depth_column, material_description_rect) in enumerate(pairs):
        for _depth_column_2, material_description_rect_2 in pairs[i + 1 :]:
            if material_description_rect.intersects(material_description_rect_2):
                to_delete.append(i)
                continue
    filtered_pairs = [item for index, item in enumerate(pairs) if index not in to_delete]

    geometric_lines = extract_lines(page, line_detection_params)

    groups = []
    if len(filtered_pairs):
        for depth_column, material_description_rect in filtered_pairs:
            description_lines = get_description_lines(lines, material_description_rect)
            if len(description_lines) > 1:
                new_groups = match_columns(
                    depth_column, description_lines, geometric_lines, material_description_rect, **params
                )
                groups.extend(new_groups)
        json_filtered_pairs = [
            {
                "depth_column": depth_column.to_json(),
                "material_description_rect": [
                    material_description_rect.x0,
                    material_description_rect.y0,
                    material_description_rect.x1,
                    material_description_rect.y1,
                ],
            }
            for depth_column, material_description_rect in filtered_pairs
        ]

    else:
        json_filtered_pairs = []
        # Fallback when no depth column was found
        material_description_rect = find_material_description_column(lines, depth_column=None)
        if material_description_rect:
            description_lines = get_description_lines(lines, material_description_rect)
            description_blocks = get_description_blocks(
                description_lines,
                geometric_lines,
                material_description_rect,
                params["block_line_ratio"],
                params["left_line_length_threshold"],
            )
            groups.extend([{"block": block} for block in description_blocks])
            json_filtered_pairs.extend(
                [
                    {
                        "depth_column": None,
                        "material_description_rect": [
                            material_description_rect.x0,
                            material_description_rect.y0,
                            material_description_rect.x1,
                            material_description_rect.y1,
                        ],
                    }
                ]
            )
    predictions = [
        {"material_description": group["block"].to_json(), "depth_interval": group["depth_interval"].to_json()}
        if "depth_interval" in group
        else {"material_description": group["block"].to_json()}
        for group in groups
    ]
    predictions = parse_and_remove_empty_predictions(predictions)
    return predictions, json_filtered_pairs


def score_column_match(
    depth_column: DepthColumn,
    material_description_rect: fitz.Rect,
    all_words: list[TextLine] | None = None,
    **params: dict,
) -> float:
    """Scores the match between a depth column and a material description.

    Args:
        depth_column (DepthColumn): The depth column.
        material_description_rect (fitz.Rect): The material description rectangle.
        all_words (list[TextLine] | None, optional): List of the available textlines. Defaults to None.
        **params (dict): Additional parameters for the matching pipeline. Kept for compatibility with the pipeline.

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
    depth_column: DepthColumn,
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    material_description_rect: fitz.Rect,
    **params: dict,
) -> list:
    """Match the depth column with the description lines.

    As part of the matching, the number of text blocks is adjusted to match the number of depth intervals.

    TODO: Check if docstring is correct.

    Args:
        depth_column (DepthColumn): The depth column.
        description_lines (list[TextLine]): The description lines.
        geometric_lines (list[Line]): The geometric lines.
        material_description_rect (fitz.Rect): The material description rectangle.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list: The matched depth intervals and text blocks.
    """
    return [
        element
        for group in depth_column.identify_groups(
            description_lines, geometric_lines, material_description_rect, **params
        )
        for element in transform_groups(group["depth_intervals"], group["blocks"], **params)
    ]


def transform_groups(
    depth_intervals: list[Interval], blocks: list[TextBlock], **params: dict
) -> list[dict[str, Interval | TextBlock]]:
    """Transforms the text blocks such that their number equals the number of depth intervals.

    If there are more depth intervals than text blocks, text blocks are splitted. When there
    are more text blocks than depth intervals, text blocks are merged. If the number of text blocks
    and depth intervals equals, we proceed with the pairing.

    Args:
        depth_intervals (List[Interval]): The depth intervals from the pdf.
        blocks (List[TextBlock]): Found textblocks from the pdf.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        List[Dict[str, Union[Interval, TextBlock]]]: Pairing of text blocks and depth intervals.
    """
    if len(depth_intervals) == 0:
        return []
    elif len(depth_intervals) == 1:
        concatenated_block = TextBlock(
            [line for block in blocks for line in block.lines]
        )  # concatenate all text lines within a block; line separation flag does not matter here.
        return [{"depth_interval": depth_intervals[0], "block": concatenated_block}]
    else:
        if len(blocks) < len(depth_intervals):
            blocks = split_blocks_by_textline_length(blocks, target_split_count=len(depth_intervals) - len(blocks))

        if len(blocks) > len(depth_intervals):
            blocks = merge_blocks_by_vertical_spacing(blocks, target_merge_count=len(blocks) - len(depth_intervals))

        return [
            {"depth_interval": depth_interval, "block": block}
            for depth_interval, block in zip(depth_intervals, blocks, strict=False)
        ]


def merge_blocks_by_vertical_spacing(blocks: list[TextBlock], target_merge_count: int) -> list[TextBlock]:
    """Merge textblocks without any geometric lines that separates them.

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

    if len(current_merged_block.lines):
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
            if len(current_block_lines):
                split_blocks.append(TextBlock(current_block_lines))
                current_block_lines = []
            if (
                block.is_terminated_by_line
            ):  # If block was terminated by a line, populate the flag to the last element of split_blocks.
                split_blocks[-1].is_terminated_by_line = True
        return split_blocks


def find_material_description_column(
    lines: list[TextLine], depth_column: DepthColumn, **params: dict
) -> fitz.Rect | None:
    """Find the material description column given a depth column.

    Args:
        lines (list[TextLine]): The text lines of the page.
        depth_column (DepthColumn): The depth column.
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

        min_y0 = max(line.rect.y0 for line in above_depth_column) if len(above_depth_column) else -1

        def check_y0_condition(y0):
            return y0 > min_y0 and y0 < depth_column.rect().y1
    else:

        def check_y0_condition(y0):
            return True

    candidate_description = [line for line in lines if check_y0_condition(line.rect.y0)]
    is_description = [line for line in candidate_description if line.is_description]

    if len(candidate_description) == 0:
        return

    description_clusters = []
    while len(is_description) > 0:
        coverage_by_generating_line = [
            [other for other in is_description if x_overlap_significant_smallest(line.rect, other.rect, 0.5)]
            for line in is_description
        ]

        def filter_coverage(coverage):
            if len(coverage):
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


def perform_matching(directory: Path, **params: dict) -> dict:
    """Perform the matching of text blocks with depth intervals.

    Args:
        directory (Path): Path to the directory that contains the pdfs.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        dict: The predictions.
    """
    for root, _dirs, files in os.walk(directory):
        output = {}
        for filename in files:
            if filename.endswith(".pdf"):
                in_path = os.path.join(root, filename)
                logger.info("Processing file: %s", in_path)
                output[filename] = {}

                with fitz.Document(in_path) as doc:
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        logger.info("Processing page %s", page_number)

                        predictions, depths_materials_column_pairs = process_page(page, **params)

                        output[filename][f"page_{page_number}"] = {
                            "layers": predictions,
                            "depths_materials_column_pairs": depths_materials_column_pairs,
                        }
        return output


if __name__ == "__main__":
    # setup mlflow tracking; should be started before any other code
    # such that tracking is enabled in other parts of the code.
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Boreholes Stratigraphy")
        mlflow.start_run()
        mlflow.log_params(flatten(line_detection_params))
        mlflow.log_params(flatten(matching_params))

    # instantiate all paths
    input_directory = DATAPATH / "Benchmark"
    ground_truth_path = input_directory / "ground_truth.json"
    out_directory = input_directory / "evaluation"
    predictions_path = input_directory / "extract" / "predictions.json"
    temp_directory = DATAPATH / "_temp"  # temporary directory to dump files for mlflow artifact logging

    # check if directories exist and create them when neccessary
    out_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    # run the matching pipeline and save the result
    predictions = perform_matching(input_directory, **matching_params)
    with open(predictions_path, "w") as file:
        file.write(json.dumps(predictions))

    # evaluate the predictions
    metrics, document_level_metrics = evaluate_matching(
        predictions_path, ground_truth_path, input_directory, out_directory
    )
    document_level_metrics.to_csv(temp_directory / "document_level_metrics.csv")  # mlflow.log_artifact expects a file

    if mlflow_tracking:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(temp_directory / "document_level_metrics.csv")
