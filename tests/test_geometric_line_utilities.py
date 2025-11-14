"""Test suite for the geometric_line_utilities module."""

import numpy as np
import pytest

from swissgeol_doc_processing.geometry.geometric_line_utilities import (
    _are_close,
    _are_mergeable,
    _are_parallel,
    _get_orthogonal_projection_to_line,
    _merge_lines,
    _odr_regression,
    is_point_near_line,
    merge_parallel_lines_quadtree,
)
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point


# Remember, phi is orthogonal to the line we are to parameterize
@pytest.fixture(
    params=[
        # Test case 1: horizontal line at y=1
        (np.array([0, 1, 2, 3]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), np.pi / 2, 1),
        # Test case 2: 45 degrees through zero
        (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), np.array([1, 1, 1, 1]), -np.pi / 4, 0),
        # Test case 3: vertical line at x=2
        (np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3]), np.array([1, 1, 1, 1]), 0, 2),
        # Test case 4: best fix is a horizontal line at y=0.5
        (np.array([0, 0, 2, 2]), np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]), np.pi / 2, 0.5),
        # Test case 4: best fix is a vertical line at x=0.5
        (np.array([0, 1, 0, 1]), np.array([0, 0, 2, 2]), np.array([1, 1, 1, 1]), 0, 0.5),
        (
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 1, 0]),
            np.array([3, 1, 1, 3]),
            np.pi / 2,
            1 / 4,
        ),  # test impact of the weights (horizontal, parallel lines)
        (
            np.array([0, 0, 1]),
            np.array([0.1, -0.1, 0]),
            np.array([2, 1, 1]),
            1.536249331404415,
            0.03362011376179206,
        ),  # test impact of the weights (three points)
    ]
)
def odr_regression_case(request):  # noqa: D103
    return request.param


# Use the fixture in the test function
def test_odr_regression(odr_regression_case):  # noqa: D103
    x, y, weights, expected_phi, expected_r = odr_regression_case
    phi, r = _odr_regression(x, y, weights)
    assert pytest.approx(expected_phi) == phi
    assert pytest.approx(expected_r) == r


@pytest.fixture(
    params=[
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),  # Test case 1 all zeroes
        (np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1])),  # Test case square
    ]
)
def odr_regression_with_zeroes_case(request):  # noqa: D103
    return request.param


def test_odr_regression_with_zeroes(odr_regression_with_zeroes_case):  # noqa: D103
    x, y = odr_regression_with_zeroes_case
    phi, r = _odr_regression(x, y, np.array([1, 1, 1, 1]))
    print(f"phi: {phi}, r: {r}")
    assert np.isnan(phi)  # Expected value
    assert np.isnan(r)  # Expected value


# Tests for the _get_orhtogonal_projection_to_line function
@pytest.fixture(
    params=[
        (Point(1, 1), 3 * np.pi / 4, 0, Point(1, 1)),  # Test case 1
        (Point(1, 0), 0, 0, Point(0, 0)),  # Test case 2
        (Point(1, 1), 0, 0, Point(0, 1)),  # Test case 3
        (Point(1, 1), np.pi / 2, 0, Point(1, 0)),  # Test case 4
        (Point(1, 1), np.pi / 2, 1, Point(1, 1)),  # Test case 5
        (Point(1, 1), np.pi / 2, -1, Point(1, -1)),  # Test case 5
        # Add more test cases here as needed
    ]
)
def orthogonal_projection_case(request):  # noqa: D103
    return request.param


def test_get_orthogonal_projection_to_line(orthogonal_projection_case):  # noqa: D103
    point, phi, r, expected_projection = orthogonal_projection_case
    projection = _get_orthogonal_projection_to_line(point, phi, r)
    assert pytest.approx(projection.tuple) == expected_projection.tuple


@pytest.fixture(
    params=[
        (
            Line(Point(0, 0), Point(1, 1)),
            Line(Point(1, 1), Point(2, 2)),
            Line(Point(0, 0), Point(2, 2)),
        ),  # 45 degrees line
        (
            Line(Point(0, 0), Point(1, 0)),
            Line(Point(1, 0), Point(2, 0)),
            Line(Point(0, 0), Point(2, 0)),
        ),  # horizontal line
        (
            Line(Point(2, 0), Point(2, 1)),
            Line(Point(2, 1), Point(2, 2)),
            Line(Point(2, 0), Point(2, 2)),
        ),  # vertical line
        (
            Line(Point(0, 0), Point(2, 0)),
            Line(Point(1, 0), Point(3, 0)),
            Line(Point(0, 0), Point(3, 0)),
        ),  # horizontal line; lines partially overlap
        (
            Line(Point(0, 0), Point(3, 0)),
            Line(Point(1, 0), Point(2, 0)),
            Line(Point(0, 0), Point(3, 0)),
        ),  # horizontal line; one line is contained in the other
        (
            Line(Point(1, 0), Point(2, 0)),
            Line(Point(0, 0), Point(3, 0)),
            Line(Point(0, 0), Point(3, 0)),
        ),  # horizontal line; one line is contained in the other (reversed)
    ]
)
def merge_lines_case(request):  # noqa: D103
    return request.param


def test_merge_lines(merge_lines_case):  # noqa: D103
    line1, line2, expected_merged_line = merge_lines_case
    merged_line = _merge_lines(line1, line2)
    assert pytest.approx(merged_line.start.tuple) == expected_merged_line.start.tuple
    assert pytest.approx(merged_line.end.tuple) == expected_merged_line.end.tuple


@pytest.mark.parametrize(
    "line1, line2, tol, expected",
    [
        # "Two lines that almost extend each other should be considered close"
        (Line(Point(0, 0), Point(0, 10)), Line(Point(0, 11), Point(0, 20)), 5, True),
        (Line(Point(0, 11), Point(0, 20)), Line(Point(0, 0), Point(0, 10)), 5, True),
        # "A line that is almost a sub-segment of another line should be considered close"
        (Line(Point(0, 0), Point(0, 20)), Line(Point(1, 8), Point(1, 12)), 5, True),
        (Line(Point(1, 8), Point(1, 12)), Line(Point(0, 0), Point(0, 20)), 5, True),
        # "Two very short parallel lines should not be considered close"
        (Line(Point(0, 0), Point(0, 0.1)), Line(Point(2, 0), Point(2, 0.1)), 5, False),
        (Line(Point(2, 0), Point(2, 0.1)), Line(Point(0, 0), Point(0, 0.1)), 5, False),
    ],
)
def test_are_close(line1, line2, tol, expected):  # noqa: D103
    assert _are_close(line1, line2, tol) == expected


def test_merge_parallel_lines_quadtree():  # noqa: D103
    lines = [
        Line(Point(0, 0), Point(1, 1)),  # line 1
        Line(Point(1, 1), Point(2, 2)),  # line 2, should be merged with line 1
        Line(Point(0, 5), Point(1, 6)),  # line 3, parallel but not close to line 1
    ]
    merged_lines = merge_parallel_lines_quadtree(lines, tol=1, angle_threshold=1)
    assert len(merged_lines) == 2, "There should be 2 lines after merging"


@pytest.mark.parametrize(
    "line, point, tol, tol_line, expected",
    [
        # Point is slightly above the line but within tol_line
        (Line(Point(0, 0), Point(10, 0)), Point(5, 2), 10, 3, True),
        # Point is too far from the line vertically
        (Line(Point(0, 0), Point(10, 0)), Point(5, 5), 10, 3, False),
        # Point is aligned with the line but outside of tol
        (Line(Point(0, 0), Point(10, 0)), Point(21, 0), 10, 3, False),
        # Point is just outside segment range, but within tol
        (Line(Point(0, 0), Point(10, 0)), Point(20, 0), 10, 3, True),
        # Tests with a diagonal line
        (Line(Point(0, 0), Point(100, 100)), Point(55.5, 55.5), 1, 1, True),
        (Line(Point(0, 0), Point(100, 100)), Point(0, 5), 1, 1, False),
        (Line(Point(0, 0), Point(100, 100)), Point(100, 95), 1, 1, False),
        (Line(Point(0, 0), Point(100, 100)), Point(100, 95), 10, 10, True),
    ],
)
def test_is_point_near_line(line, point, tol, tol_line, expected):  # noqa: D103
    assert is_point_near_line(line, point, segment_extension_tol=tol, perpendicular_tol=tol_line) == expected


@pytest.mark.parametrize(
    "line1, line2, expected",
    [
        # Two vertical lines (x = constant), nearly parallel
        (Line(Point(0, 0), Point(0, 10)), Line(Point(1e-6, 0), Point(1e-6, 10)), True),
        # Two vertical lines, one with opposite direction, still parallel
        (Line(Point(0, 0), Point(0, 10)), Line(Point(0, 10), Point(0, 0)), True),
        # Slightly off vertical (shows tan discontinuity), this fails if the 2nd condition in are_parallel is not there
        (Line(Point(0, 0), Point(1e-6, 10)), Line(Point(0, 0), Point(-1e-6, 10)), True),
        # One vertical, one not quite vertical – angle should exceed threshold
        (Line(Point(0, 0), Point(0, 10)), Line(Point(0, 0), Point(1, 10)), False),
    ],
)
def test_are_parallel_vertical_cases(line1, line2, expected):  # noqa: D103
    angle_threshold = 5.0  # degrees

    result = _are_parallel(line1, line2, angle_threshold)
    assert result == expected, f"Expected {expected} for lines {line1} and {line2}, got {result}"


@pytest.mark.parametrize(
    "line1, line2, should_be_merge",
    [
        (
            Line(Point(0, 0), Point(2, 2)),  # not parallel
            Line(Point(0, 1), Point(2, 2.5)),
            False,
        ),
        (
            Line(Point(0, 0), Point(0, 2)),  # |  dist=1, but next to each other, more restrictive
            Line(Point(1, 0), Point(1, 1)),  # ||
            False,
        ),
        (
            Line(Point(0, 0), Point(0, 1)),  #  | dist=1, but following eachother, within tolerance
            Line(Point(0, 2), Point(0, 3)),
            True,  #                            |
        ),
        (
            Line(Point(0, 0), Point(0, 2)),  #     |      next to each other, but much closer
            Line(Point(0.1, 0), Point(0.1, 1)),  # ||
            True,
        ),
    ],
)
def test_are_mergeable(line1, line2, should_be_merge):
    """Test the _are_mergeable function with various line pairs.

    The lines should be mergeable if they are parallel and close enough to each other. The tolored merging distance
    is smaller for lines that are next to each other than for lines that follow each other.

    Args:
        line1 (Line): The first line to test.
        line2 (Line): The second line to test.
        should_be_merge (bool): Expected result of the mergeability check.
    """
    result = _are_mergeable(line1, line2, tol=1.1, angle_threshold=5)
    assert result == should_be_merge, f"Expected {should_be_merge} for lines {line1} and {line2}, got {result}"


@pytest.mark.parametrize(
    "line, point, expected_distance",
    [
        # Point on horizontal line segment
        (Line(Point(0, 0), Point(10, 0)), Point(5, 0), 0.0),
        # Point directly above horizontal line segment (perpendicular distance)
        (Line(Point(0, 0), Point(10, 0)), Point(5, 3), 3.0),
        # Point at start of line segment
        (Line(Point(0, 0), Point(10, 0)), Point(0, 0), 0.0),
        # Point at end of line segment
        (Line(Point(0, 0), Point(10, 0)), Point(10, 0), 0.0),
        # Point before start of segment (should clamp to start)
        (Line(Point(0, 0), Point(10, 0)), Point(-5, 0), 5.0),
        # Point after end of segment (should clamp to end)
        (Line(Point(0, 0), Point(10, 0)), Point(15, 0), 5.0),
        # Vertical line segment
        (Line(Point(0, 0), Point(0, 10)), Point(0, 5), 0.0),
    ],
)
def test_distance_to_segment(line, point, expected_distance):
    """Test the distance_to_segment method of the Line class."""
    result = line.distance_to_segment(point)
    if isinstance(expected_distance, float):
        assert pytest.approx(result, abs=0.001) == expected_distance
    else:
        assert result == expected_distance


@pytest.mark.parametrize(
    "line, point, threshold, expected",
    [
        # Point on the line segment
        (Line(Point(0, 0), Point(10, 0)), Point(5, 0), 1.0, True),
        # Point just within threshold
        (Line(Point(0, 0), Point(10, 0)), Point(5, 1), 1.0, True),
        # Point exactly at threshold
        (Line(Point(0, 0), Point(10, 0)), Point(5, 1), 1.0, True),
        # Point just outside threshold
        (Line(Point(0, 0), Point(10, 0)), Point(5, 1.1), 1.0, False),
        # Point near end of segment
        (Line(Point(0, 0), Point(10, 0)), Point(10, 0.5), 1.0, True),
        # Point beyond segment but within threshold of endpoint
        (Line(Point(0, 0), Point(10, 0)), Point(11, 0), 1.5, True),
        # Point beyond segment and outside threshold
        (Line(Point(0, 0), Point(10, 0)), Point(15, 0), 1.0, False),
    ],
)
def test_point_near_segment(line, point, threshold, expected):
    """Test the point_near_segment method of the Line class."""
    result = line.point_near_segment(point, threshold)
    assert result == expected


@pytest.mark.parametrize(
    "line1, line2, expected",
    [
        # Intersection at center
        (Line(Point(0, 0), Point(10, 10)), Line(Point(0, 10), Point(10, 0)), True),
        # Horizontal and vertical lines intersecting
        (Line(Point(0, 5), Point(10, 5)), Line(Point(5, 0), Point(5, 10)), True),
        # Lines that don't intersect (parallel)
        (Line(Point(0, 0), Point(10, 0)), Line(Point(0, 1), Point(10, 1)), False),
        # Lines that don't intersect (would intersect if extended)
        (Line(Point(0, 0), Point(5, 0)), Line(Point(10, -5), Point(10, 5)), False),
        # Lines that share an endpoint
        (Line(Point(0, 0), Point(5, 5)), Line(Point(5, 5), Point(10, 0)), True),
        # Lines that share a starting point
        (Line(Point(0, 0), Point(5, 5)), Line(Point(0, 0), Point(5, -5)), True),
        # T-intersection (one line ends on another)
        (Line(Point(0, 0), Point(10, 0)), Line(Point(5, -5), Point(5, 0)), True),
        # Lines very close but don't intersect
        (Line(Point(0, 0), Point(10, 0)), Line(Point(0, 0.1), Point(10, 0.1)), False),
    ],
)
def test_intersects_with(line1, line2, expected):
    """Test the intersects_with method of the Line class."""
    result = line1.intersects_with(line2)
    assert result == expected
    reverse_result = line2.intersects_with(line1)
    assert result == reverse_result, f"Intersection should be symmetric: {line1} with {line2}"
