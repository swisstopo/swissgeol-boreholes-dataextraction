"""Test suite for the dataclasses module."""

import pytest

from extraction.features.utils.geometry.geometry_dataclasses import Line, Point


def test_distance_to():  # noqa: D103
    line = Line(Point(0, 0), Point(0, 1))
    point = Point(1, 0)
    distance = line.distance_to(point)
    assert pytest.approx(distance) == 1, "The distance should be approximately 1"


def test_slope():  # noqa: D103
    line = Line(Point(0, 0), Point(1, 1))
    assert line.slope == 1, "The slope should be 1"


def test_intercept():  # noqa: D103
    line = Line(Point(0, 0), Point(1, 1))
    assert line.intercept == 0, "The intercept should be 0"
