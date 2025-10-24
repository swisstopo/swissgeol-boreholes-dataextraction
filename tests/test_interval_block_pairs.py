"""Unit tests for interval-to-block pair matching using dynamic programming.

This module tests the functionality of matching text lines to intervals in a sidebar
using a dynamic programming approach. It verifies the correct pairing of text blocks
with their corresponding depth intervals based on spatial relationships and affinity scores.
"""

import pymupdf
import pytest

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import AAboveBSidebar
from extraction.features.utils.text.textline import TextLine, TextWord
from utils.dynamic_matching import IntervalToLinesDP


@pytest.fixture
def a_above_b_sidebar():
    """Create a fixture for testing with three depth column entries.

    Returns:
        AAboveBSidebar: A sidebar with three depth entries positioned at
            y-coordinates 0-10, 20-30, and 40-50, all with x-coordinates 0-30.
    """
    x0 = 0.0
    x1 = 30.0
    return AAboveBSidebar(
        [
            DepthColumnEntry(0.0, pymupdf.Rect(x0, 0, x1, 10), 0.0),
            DepthColumnEntry(0.0, pymupdf.Rect(x0, 20, x1, 30), 0.0),
            DepthColumnEntry(0.0, pymupdf.Rect(x0, 40, x1, 50), 0.0),
        ]
    )


@pytest.fixture
def lines():
    """Create a fixture with three text lines for testing.

    Returns:
        list[TextLine]: Three text lines positioned at y-coordinates 11-14,
            15-19, and 31-39, all with x-coordinates 40-60.
    """
    x0 = 40
    x1 = 60
    return [
        TextLine([TextWord(pymupdf.Rect(x0, 11, x1, 14), "", 0.0)]),
        TextLine([TextWord(pymupdf.Rect(x0, 15, x1, 19), "", 0.0)]),
        TextLine([TextWord(pymupdf.Rect(x0, 31, x1, 39), "", 0.0)]),
    ]


@pytest.mark.parametrize(
    "affinity,expected_mapping",
    [
        pytest.param([0.0, 0.0, 0.0], [(0, [0, 1]), (1, [2])], id="standard_map"),
        pytest.param([0.0, -1.0, 0.0], [(0, [0]), (1, [1, 2])], id="lines_not_compatible"),
    ],
)
def test_dp_manual(a_above_b_sidebar, lines, affinity, expected_mapping):
    """Test dynamic programming matching with manual affinity scores.

    Args:
        a_above_b_sidebar: Fixture containing three depth intervals.
        lines: Fixture containing three text lines.
        affinity: List of affinity scores between intervals and lines.
        expected_mapping: Expected grouping of lines to intervals.

    Tests two scenarios:
        1. Standard mapping where first two lines map to first interval
        2. Modified mapping when middle line has negative affinity with the one above.
    """
    zones = a_above_b_sidebar.get_interval_zone()
    dp = IntervalToLinesDP(zones, lines, affinity)  # first 0.0 default
    _, mapping = dp.solve(a_above_b_sidebar.dp_scoring_fn)

    assert mapping == [(zones[zone_idx], [lines[i] for i in line_idxs]) for zone_idx, line_idxs in expected_mapping]
