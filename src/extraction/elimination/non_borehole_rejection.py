"""Module for detecting and rejecting non-borehole pages based on text content."""

import string

import numpy as np
import pymupdf
from shapely.geometry import box
from shapely.ops import unary_union

from extraction.features.utils.text.textblock import TextBlock
from extraction.features.utils.text.textline import TextLine
from utils.file_utils import read_params

matching_params = read_params("matching_params.yml")


def overlaps(line, line2) -> bool:
    """Check if two text lines overlap."""
    vertical_margin = 15
    ref_rect = pymupdf.Rect(
        line.rect.x0,
        line.rect.y0 - vertical_margin,
        line.rect.x1,
        line.rect.y1 + vertical_margin,
    )
    return ref_rect.intersects(line2.rect)


def adjacent_lines(lines: list[TextLine]) -> list[set[int]]:
    """Find adjacent lines that overlap."""
    result = [set() for _ in lines]
    for index, line in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index2 > index and overlaps(line, line2):
                result[index].add(index2)
                result[index2].add(index)
    return result


def apply_transitive_closure(data: list[set[int]]) -> bool:
    """Apply transitive closure to the adjacency list."""
    found_new_relation = False
    for index, adjacent_indices in enumerate(data):
        new_adjacent_indices = set()
        for adjacent_index in adjacent_indices:
            new_adjacent_indices.update(
                new_index for new_index in data[adjacent_index] if new_index not in data[index]
            )

        for new_adjacent_index in new_adjacent_indices:
            data[index].add(new_adjacent_index)
            data[new_adjacent_index].add(index)
            found_new_relation = True
    return found_new_relation


def create_text_blocks(text_lines: list[TextLine]) -> list[TextBlock]:
    """Sort lines into TextBlocks."""
    data = adjacent_lines(text_lines)

    while apply_transitive_closure(data):
        pass

    blocks: list[TextBlock] = []
    remaining_indices = {index for index, _ in enumerate(data)}
    for index, adjacent_indices in enumerate(data):
        if index in remaining_indices:
            selected_indices = adjacent_indices
            selected_indices.add(index)
            blocks.append(TextBlock([text_lines[selected_index] for selected_index in sorted(list(selected_indices))]))
            remaining_indices.difference_update(selected_indices)

    return blocks


def get_union_areas(rects: list[pymupdf.Rect]) -> float:
    """Compute total non-overlapping area from list of bounding box rects."""
    shapes = [box(rect.x0, rect.y0, rect.x1, rect.y1) for rect in rects]
    return unary_union(shapes).area if shapes else 0


def is_borehole_page(text_lines: list[TextLine], language: str) -> bool:
    """Determine if a page is a borehole page based on text content.

    Args:
        text_lines (list[TextLine]): The text lines extracted from the page.
        language (str): The language of the text content.

    Returns:
        bool: True if the page is a borehole page, False otherwise.
    """
    if not text_lines:
        return False
    words_per_line = [len(line.words) for line in text_lines]
    mean_words_per_line = np.mean(words_per_line)

    text_blocks = create_text_blocks(text_lines)
    block_union = get_union_areas([block.rect for block in text_blocks])
    if block_union == 0:
        return False
    word_union = get_union_areas(
        [word.rect for block in text_blocks for line in block.lines for word in line.words if len(line.words) > 1]
    )
    word_density = word_union / block_union

    all_text_words = [word for line in text_lines for word in line.words]
    total_words = 0
    material_words = 0
    for word in all_text_words:
        if not word.text.strip(string.punctuation).isalpha():
            continue
        total_words += 1
        if TextLine([word]).is_description(matching_params["material_description"], language):
            material_words += 1
    material_words_ratio = material_words / total_words if total_words else 0.0

    # Lillemor's rule
    # return not (word_density > 0.5 and mean_words_per_line > 3 and material_words_ratio < 0.015)

    # Thresholds obtained by fitting a decision tree and simplifying the classification rules
    return (material_words_ratio <= 0.06 and (mean_words_per_line <= 2.08 or word_density <= 0.39)) or (
        material_words_ratio > 0.06 and mean_words_per_line >= 1.20
    )
