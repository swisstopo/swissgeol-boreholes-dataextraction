"""Methods for finding the best grouping of material description lines in the absence of a sidebar."""

from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_affinity import Affinity


def get_descriptions_blocks(
    description_lines: list[TextLine], affinities: list[Affinity], no_sidebar_weights: dict[str, float]
) -> list[TextBlock]:
    """Based on the line affinity, group the description lines into blocks.

    The grouping is done based on the presence of geometric lines, the indentation of lines
    and the vertical spacing between lines.

    Args:
        description_lines (list[TextLine]): The text lines to group into blocks.
        affinities (list[Affinity]): the affinity between each line pair, previously computed.
        no_sidebar_weights (dict): the matching parameters.

    Returns:
        list[TextBlock]: A list of description lines grouped into text blocks
    """
    blocks = []
    prev_block_idx = 0

    horizontal_lines_significance = sum(-affinity.long_lines_affinity for affinity in affinities)
    vertical_spacing_significance = sum(max(0.0, -affinity.vertical_spacing_affinity) for affinity in affinities)
    if horizontal_lines_significance > vertical_spacing_significance:
        # if the presence of horizontal lines seems to be a stronger differentiator than the vertical spacing between
        # lines, then we set the threshold at the level of the presence of such a horizontal line, making it less
        # likely that descriptions are split in the absence of such a line.
        threshold = -no_sidebar_weights["line_weight"]
    else:
        threshold = -0.2 * no_sidebar_weights["spacing_weight"]

    for line_idx, affinity in enumerate(affinities):
        # note: the affinity of the first line is always 0.0
        if affinity.weighted_affinity(**no_sidebar_weights) <= threshold:
            blocks.append(TextBlock(description_lines[prev_block_idx:line_idx]))
            prev_block_idx = line_idx

    blocks.append(TextBlock(description_lines[prev_block_idx:]))
    return blocks
