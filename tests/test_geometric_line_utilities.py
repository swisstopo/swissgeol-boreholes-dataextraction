"""Test suite for the geometric_line_utilities module."""

import numpy as np
import pytest
from stratigraphy.util.dataclasses import Line, Point
from stratigraphy.util.geometric_line_utilities import (
    _get_orthogonal_projection_to_line,
    _merge_lines,
    _odr_regression,
)


# Remember, phi is orthogonal to the line we are to parameterize
# The way phi is defined is a bit counterintuitive, but it seems consistent all along.
@pytest.fixture(
    params=[
        (np.array([0, 1, 2, 3]), np.array([1, 1, 1, 1]), -np.pi / 2, 1),  # Test case 1 horizontal line at y=0
        (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), -np.pi / 4, 0),  # Test case 2 45 degrees through zero
        (np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3]), 0, 2),  # Test case 3 vertical line at x=2
        # Add more test cases here as needed
    ]
)
def odr_regression_case(request):  # noqa: D103
    return request.param


# Use the fixture in the test function
def test_odr_regression(odr_regression_case):  # noqa: D103
    x, y, expected_phi, expected_r = odr_regression_case
    phi, r = _odr_regression(x, y, np.array([1, 1, 1, 1]))
    assert pytest.approx(np.sin(expected_phi)) == np.sin(phi)
    assert pytest.approx(np.abs(expected_r)) == np.abs(r)


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
        # Add more test cases here as needed
    ]
)
def orthogonal_projection_case(request):  # noqa: D103
    return request.param


def test_get_orthogonal_projection_to_line(orthogonal_projection_case):  # noqa: D103
    point, phi, r, expected_projection = orthogonal_projection_case
    projection = _get_orthogonal_projection_to_line(point, phi, r)
    assert pytest.approx(projection) == expected_projection


@pytest.fixture(
    params=[
        # Assuming Line takes two points as arguments
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
        # Add more test cases here as needed
    ]
)
def merge_lines_case(request):  # noqa: D103
    return request.param


def test_merge_lines(merge_lines_case):  # noqa: D103
    line1, line2, expected_merged_line = merge_lines_case
    merged_line = _merge_lines(line1, line2)
    assert (
        pytest.approx(merged_line.start.tuple) == expected_merged_line.start.tuple
        and pytest.approx(merged_line.end.tuple) == expected_merged_line.end.tuple
    )  # Adjust this line if Line objects can't be compared directly
