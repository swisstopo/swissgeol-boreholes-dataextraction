"""Methods for finding the best correspondence between depth intervals and material description lines."""

from extraction.features.stratigraphy.interval.interval import IntervalBlockPair, IntervalZone
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import AAboveBSidebar
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import ProtocolSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from extraction.utils.dynamic_matching import IntervalToLinesDP
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_affinity import Affinity


def match_with_dynamic_programming(
    sidebar: Sidebar, description_lines: list[TextLine], affinity_scores: list[float]
) -> list[tuple[IntervalZone, list[TextLine]]]:
    """Apply dynamic programming to find the best matching between depth intervals and description lines.

    Args:
        sidebar (Sidebar): the sidebar to match the descriptions with.
        description_lines (list[TextLine]): The text lines that to create the material descriptions from.
        affinity_scores (list[float]): the affinity score for each line with the one above, previously computed and
            weighted according to the sidebar type.

    Returns:
        list[tuple[IntervalZone, list[TextLine]]]: The computed mapping between depth intervals and description lines.
    """
    depth_interval_zones = sidebar.get_interval_zone()

    dp = IntervalToLinesDP(depth_interval_zones, description_lines, affinity_scores)

    _, mapping = dp.solve(sidebar.dp_scoring_fn)

    # Did we add an implicit depth interval at the start (interval.start = None) and this was not assigned to any
    # material description? In this case, we remove it again.
    if len(mapping):
        interval, text_lines = mapping[0]
        if interval.start is None and not text_lines:
            mapping.pop(0)

    return mapping


def partition_lines(description_lines: list[TextLine]) -> list[list[TextLine]]:
    """Partition the description lines, separating where the vertical gap between two consecutive lines is large.

    A new partition is started wherever the vertical gap between two consecutive lines is greater than the maximum
    of the heights of both lines.

    Args:
        description_lines (list[TextLine]): The description lines to partition.

    Returns:
        list[list[TextLine]]: The computed partitions of the description lines.
    """
    partitions = []
    current_partition = []
    for line in description_lines:
        if current_partition:
            prev_line = current_partition[-1]
            gap = line.rect.y0 - prev_line.rect.y1
            if gap > max(prev_line.rect.height, line.rect.height):
                partitions.append(current_partition)
                current_partition = []
        current_partition.append(line)
    if current_partition:
        partitions.append(current_partition)
    return partitions


def weight_affinity_scores(affinity_scores: list[float], description_lines: list[TextLine]) -> list[float]:
    """Weight the affinity scores linearly by the number of lines in the partition each line belongs to.

    The partitions are computed from the vertical gaps between consecutive description lines (see partition_lines).
    The affinity score of each line is multiplied by the number of lines in its partition, so that affinities within
    larger (more significant) partitions have a stronger influence on the dynamic programming matching.

    Args:
        affinity_scores (list[float]): The affinity scores, one per description line.
        description_lines (list[TextLine]): The description lines corresponding to the affinity scores.

    Returns:
        list[float]: The reweighted affinity scores.
    """
    partitions = partition_lines(description_lines)
    # partitions preserve the order of the description lines, so the sizes are aligned with the affinity scores
    partition_sizes = [len(partition) for partition in partitions for _ in partition]
    return [score * size / 10 for score, size in zip(affinity_scores, partition_sizes, strict=True)]


def is_great_protocol_sidebar_mapping(mapping: list[tuple[IntervalZone, list[TextLine]]], threshold: float) -> bool:
    """Check for a great protocol sidebar fit: depths vertically aligned with the first description lines.

    Args:
        mapping (list[tuple[IntervalZone, list[TextLine]]]): a mapping between depth intervals and description lines.
        threshold (float): a great fit must have an average vertical offset that is below this threshold.

    Returns:
        bool: True if the provided mapping matches the expected layout of a protocol sidebar very precisely
    """
    total_relative_y_offset = 0
    for interval, lines in mapping:
        if interval.start is None or not lines:
            return False
        total_relative_y_offset += abs((interval.start.y0 - lines[0].rect.y0) / interval.start.height)

    return total_relative_y_offset / len(mapping) < threshold


def match_lines_to_interval(
    sidebar: Sidebar,
    description_lines: list[TextLine],
    affinities: list[Affinity],
    diagonals: list[Line],
    protocol_great_match_threshold: float,
) -> list[IntervalBlockPair]:
    """Match the description lines to the pair intervals.

    Args:
        sidebar (Sidebar): The sidebar.
        description_lines (list[TextLine]): The description lines.
        affinities (list[Affinity]): the affinity between each line pair, previously computed.
        diagonals (list[Line]): The diagonal lines linking text lines to intervals.
        protocol_great_match_threshold (float): Threshold for checking if a mapping is a great protocol sidebar fit.

    Returns:
        list[IntervalBlockPair]: The matched depth intervals and text blocks.
    """
    # shift the entries of the sidebar using the diagonals, only relevant for AAboveBSidebars
    if isinstance(sidebar, AAboveBSidebar):
        # If everything fits really well as a protocol sidebar, then we should treat it as one
        protocol_sidebar = ProtocolSidebar(sidebar.unfiltered_entries)
        mapping = match_with_dynamic_programming(
            protocol_sidebar, description_lines, protocol_sidebar.dp_weighted_affinities(affinities)
        )
        if is_great_protocol_sidebar_mapping(mapping, protocol_great_match_threshold):
            return protocol_sidebar.post_processing(mapping)

        # Adjust coordinates based on the presence of diagonal lines
        sidebar.compute_entries_shift(diagonals)
        sidebar.prevent_shifts_crossing()

    # affinities within larger partitions (separated by large vertical gaps) are given more weight
    weighted_affinities = weight_affinity_scores(sidebar.dp_weighted_affinities(affinities), description_lines)

    mapping = match_with_dynamic_programming(sidebar, description_lines, weighted_affinities)

    for entry in sidebar.entries:
        # Reset for when the same entries are re-used again in the context of a different sidebar / material
        # description rect.
        # TODO: properly decouple the relative shift from the actual (ideally immutable) depth column entry.
        entry.relative_shift = 0.0

    return sidebar.post_processing(mapping)
