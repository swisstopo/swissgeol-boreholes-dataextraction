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
    sidebar: Sidebar, description_lines: list[TextLine], affinities: list[Affinity]
) -> list[tuple[IntervalZone, list[TextLine]]]:
    """Apply dynamic programming to find the best matching between depth intervals and description lines.

    Args:
        sidebar (Sidebar): the sidebar to match the descriptions with.
        description_lines (list[TextLine]): The text lines that to create the material descriptions from.
        affinities (list[Affinity]): the affinity between each line pair, previously computed.

    Returns:
        list[tuple[IntervalZone, list[TextLine]]]: The computed mapping between depth intervals and description lines.
    """
    depth_interval_zones = sidebar.get_interval_zone()

    # affinities can differ depending on sidebar type
    affinity_scores = sidebar.dp_weighted_affinities(affinities)
    dp = IntervalToLinesDP(depth_interval_zones, description_lines, affinity_scores)

    _, mapping = dp.solve(sidebar.dp_scoring_fn)
    return mapping


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
        mapping = match_with_dynamic_programming(protocol_sidebar, description_lines, affinities)
        if is_great_protocol_sidebar_mapping(mapping, protocol_great_match_threshold):
            return protocol_sidebar.post_processing(mapping)

        # Adjust coordinates based on the presence of diagonal lines
        sidebar.compute_entries_shift(diagonals)
        sidebar.prevent_shifts_crossing()

    mapping = match_with_dynamic_programming(sidebar, description_lines, affinities)
    return sidebar.post_processing(mapping)
