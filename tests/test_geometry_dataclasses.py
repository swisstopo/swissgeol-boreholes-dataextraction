"""Test suite for the geometry_dataclasses module."""

import pytest

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point


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


@pytest.mark.parametrize(
    "line, y, expected_x",
    [
        # Diagonal line
        (Line(Point(0, 0), Point(1, 1)), 0, 0),
        (Line(Point(0, 0), Point(1, 1)), 1, 1),
        (Line(Point(0, 0), Point(1, 1)), 2, 2),
        (Line(Point(0, 0), Point(1, 1)), -2, -2),
        # Horizontal line
        (Line(Point(0, 1), Point(1, 1)), 0, None),
        (Line(Point(0, 1), Point(1, 1)), 1, None),
        (Line(Point(0, 1), Point(1, 1)), 2, None),
        # Vertical line
        (Line(Point(1, 0), Point(1, 1)), 0, 1),
        (Line(Point(1, 0), Point(1, 1)), 1, 1),
        (Line(Point(1, 0), Point(1, 1)), 2, 1),
    ],
)
def test_xfromy(line, y, expected_x):
    """Test the intersects_with method of the Line class."""
    assert line.x_from_y(y) == expected_x
