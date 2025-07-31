"""Tests for the resolve_reference function in data_formatter module."""

import pytest

from classification.utils.data_formatter import resolve_reference


@pytest.fixture
def previous_layers():  # noqa: D103
    return [
        {"material_description": "Sand, brown, medium-grained", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "Clay, grey, soft", "depth_interval": {"start": 2.0, "end": 3.0}},
        {"material_description": "Gravel, coarse", "depth_interval": {"start": 3.0, "end": 4.0}},
    ]


@pytest.mark.parametrize(
    "description,expected",
    [
        ("Silt, dark grey", "Silt, dark grey"),
        ("Same as 2.0m, but darker", "Clay, grey, soft, but darker"),
        ("Do but finer.", "Gravel, coarse but finer."),
        ("Dolomite, light grey", "Dolomite, light grey"),
        ("Dirt with 1.0-2.0m depth", "Dirt with 1.0-2.0m depth"),
    ],
)
def test_resolve_reference(description, expected, previous_layers):
    """Test reference resolution with various scenarios."""
    result = resolve_reference(description, previous_layers)
    assert result == expected
