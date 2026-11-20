"""Regression tests for matching diagonal layer boundaries to depth labels."""

import json
from decimal import Decimal
from pathlib import Path

import pymupdf
import pytest

from extraction.features.stratigraphy.depth_description_alignment import match_lines_to_interval
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import AAboveBSidebar
from extraction.features.stratigraphy.sidebarentry.depth_column_entry import DepthColumnEntry
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from swissgeol_doc_processing.text.textline_affinity import Affinity


@pytest.fixture
def a359_alignment():
    """Load the observed sidebar, connectors and description lines of A359 page 1."""
    data = json.loads((Path(__file__).parent / "fixtures/a359_diagonal_alignment.json").read_text())
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(*rect), value=Decimal(depth), page_number=1)
            for depth, *rect in data["entries"]
        ]
    )
    diagonals = [Line(Point(x0, y0), Point(x1, y1)) for x0, y0, x1, y1 in data["diagonals"]]
    lines = [TextLine([TextWord(pymupdf.Rect(*rect), text, 1)]) for text, *rect in data["lines"]]
    affinities = [Affinity(*values) for values in data["affinities"]]
    return sidebar, diagonals, lines, affinities


def test_a359_complete_descriptions_follow_diagonal_boundaries(a359_alignment):
    """Keep both lines of each paragraph inside its physical depth interval."""
    sidebar, diagonals, lines, affinities = a359_alignment
    pairs = match_lines_to_interval(sidebar, lines, affinities, diagonals, 0.1)
    descriptions = {
        (pair.depth_interval.start.value if pair.depth_interval.start else None, pair.depth_interval.end.value): [
            line.text for line in pair.block.lines
        ]
        for pair in pairs
        if pair.depth_interval.end
    }
    assert descriptions[Decimal("0.1"), Decimal("0.6")] == ["line_01", "line_02"]
    assert descriptions[Decimal("3.2"), Decimal("5.8")] == ["line_12", "line_13"]
    assert descriptions[Decimal("13.05"), Decimal("16")] == ["line_35", "line_36"]


def test_a359_depth_labels_match_ordered_connector_endpoints(a359_alignment):
    """Resolve nearby competing labels in order, including labels above a connector."""
    sidebar, diagonals, _, _ = a359_alignment
    sidebar.compute_entries_shift(list(reversed(diagonals)))
    by_depth = {entry.value: entry for entry in sidebar.entries}
    for depth, endpoint in [
        ("3.2", 202.28147888183594),
        ("5.8", 217.65386962890625),
        ("13.05", 399.45953369140625),
        ("16", 417.25909423828125),
    ]:
        rect = by_depth[Decimal(depth)].shifted_rect
        assert (rect.y0 + rect.y1) / 2 == pytest.approx(endpoint)
    assert by_depth[Decimal("19")].relative_shift == 0


@pytest.mark.parametrize("entry_ys, diagonal_ys", [([], []), ([0], []), ([], [0]), ([0], [100])])
def test_missing_or_remote_diagonals_leave_entries_unshifted(entry_ys, diagonal_ys):
    """Do not force a match when a label or nearby connector is missing."""
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(0, y, 5, y + 8), value=Decimal(index), page_number=1)
            for index, y in enumerate(entry_ys)
        ]
    )
    sidebar.compute_entries_shift([Line(Point(10, y), Point(20, y + 20)) for y in diagonal_ys])
    assert all(entry.relative_shift == 0 for entry in sidebar.entries)


def test_missing_connector_and_duplicate_do_not_steal_later_match():
    """Skip a missing middle connector and noise while preserving subsequent matches."""
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(0, y, 5, y + 8), value=Decimal(index), page_number=1)
            for index, y in enumerate([0, 20, 40])
        ]
    )
    diagonal = Line(Point(10, 11), Point(20, 30))
    sidebar.compute_entries_shift(
        [
            Line(Point(10, 100), Point(20, 120)),
            Line(Point(10, 44), Point(20, 34)),
            diagonal,
            diagonal,
        ]
    )
    assert [entry.relative_shift for entry in sidebar.entries] == [26, 0, -10]


def test_zero_height_labels_are_not_shifted():
    """Ignore degenerate OCR boxes rather than divide by zero."""
    sidebar = AAboveBSidebar([DepthColumnEntry(rect=pymupdf.Rect(0, 4, 5, 4), value=Decimal(1), page_number=1)])
    sidebar.compute_entries_shift([Line(Point(10, 4), Point(20, 20))])
    assert sidebar.entries[0].relative_shift == 0


def test_competing_connectors_inside_label_prefer_closest_center():
    """Distinguish nearby connectors even when all start inside the same OCR box."""
    sidebar = AAboveBSidebar([DepthColumnEntry(rect=pymupdf.Rect(0, 0, 5, 8), value=Decimal(1), page_number=1)])
    sidebar.compute_entries_shift(
        [
            Line(Point(10, 2), Point(20, 10)),
            Line(Point(10, 4), Point(20, 20)),
            Line(Point(10, 6), Point(20, 30)),
        ]
    )
    assert sidebar.entries[0].relative_shift == 16


def test_duplicate_connector_cannot_fill_missing_middle_boundary():
    """Use one physical connector once even if it occurs twice in the input."""
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(0, y, 5, y + 8), value=Decimal(index), page_number=1)
            for index, y in enumerate([0, 20, 40])
        ]
    )
    diagonal = Line(Point(10, 12), Point(20, 30))
    sidebar.compute_entries_shift([diagonal, diagonal, Line(Point(10, 44), Point(20, 60))])
    assert [entry.relative_shift for entry in sidebar.entries] == [26, 0, 16]


def test_near_duplicate_connectors_are_merged_before_alignment():
    """Respect the existing line detector's merging before matching nearby connectors."""
    from swissgeol_doc_processing.geometry.geometric_line_utilities import merge_parallel_lines_quadtree
    from swissgeol_doc_processing.utils.file_utils import read_params

    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(0, y, 5, y + 8), value=Decimal(index), page_number=1)
            for index, y in enumerate([0, 20, 40])
        ]
    )
    params = read_params("line_detection_params.yml")["line_merging_params"]
    diagonals = merge_parallel_lines_quadtree(
        [
            Line(Point(10, 11.8), Point(20, 30)),
            Line(Point(10, 12.2), Point(20, 30.2)),
            Line(Point(10, 44), Point(20, 60)),
        ],
        tol=params["merging_tolerance"],
        angle_threshold=params["angle_threshold"],
    )
    assert len(diagonals) == 2
    sidebar.compute_entries_shift(diagonals)
    assert sidebar.entries[0].relative_shift == pytest.approx(26.177019229100523)
    assert sidebar.entries[1].relative_shift == 0
    assert sidebar.entries[2].relative_shift == 16
