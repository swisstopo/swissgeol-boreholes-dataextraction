"""Test suite for the coordinate_extraction module."""

import fitz
import pytest
from stratigraphy import DATAPATH
from stratigraphy.util.coordinate_extraction import (
    Coordinate,
    CoordinateEntry,
    CoordinateExtractor,
    LV03Coordinate,
    LV95Coordinate,
)


def test_reprLV95():  # noqa: D103
    coord = LV95Coordinate(CoordinateEntry(2789456), CoordinateEntry(1123012))
    assert repr(coord) == "E: 2'789'456, N: 1'123'012"


def test_to_jsonLV95():  # noqa: D103
    coord = LV95Coordinate(CoordinateEntry(2789456), CoordinateEntry(1123012))
    assert coord.to_json() == {
        "E": 2789456,
        "N": 1123012,
    }


def test_swap_coordinates():  # noqa: D103
    north = CoordinateEntry(789456)
    east = CoordinateEntry(123012)
    coord = LV95Coordinate(north=north, east=east)
    assert coord.east == north
    assert coord.north == east


def test_reprLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry(789456), CoordinateEntry(123012))
    assert repr(coord) == "E: 789'456, N: 123'012"


def test_to_jsonLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry(789456), CoordinateEntry(123012))
    assert coord.to_json() == {
        "E": 789456,
        "N": 123012,
    }


doc = fitz.open(DATAPATH.parent / "example" / "example_borehole_profile.pdf")
extractor = CoordinateExtractor(doc)


def test_CoordinateExtractor_extract_coordinates():  # noqa: D103
    # Assuming there is a method called 'extract' in CoordinateExtractor class
    coordinates = extractor.extract_coordinates()
    # Check if the returned value is a list
    assert isinstance(coordinates, Coordinate)
    assert repr(coordinates.east) == "615'790"
    assert repr(coordinates.north) == "157'500"


def test_CoordinateExtractor_find_coordinate_key():  # noqa: D103
    text = "This is a sample text followed by a key with a spelling mistake Ko0rdinate 615.790 / 157.500"
    key = extractor.find_coordinate_key(text)
    assert key == "Ko0rdinate "


def test_CoordinateExtractor_get_coordinate_substring():  # noqa: D103
    text = (
        "This is a sample text followed by a key with a spelling"
        "mistake Ko0rdinate and some noise 615.79o /\n157; 500 in the middle."
    )
    substring = extractor.get_coordinate_substring(text)
    assert substring == "and s0me n0ise 615.790 / 157; 500 in the middle."


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate 615.790 / 157.500 and some noise",
            (615790, 157500),
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X= 615.790 / Y157.500 and some noise",
            (615790, 157500),
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X: 2'615'790 / 1'157'500 and some noise",
            (2615790, 1157500),
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X 2615790 / 1157500 and some noise",
            (2615790, 1157500),
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate 615790 / 157500 and some noise",
            (615790, 157500),
        ),
    ],
)
def test_CoordinateExtractor_get_coordinate_pairs(text, expected):  # noqa: D103
    coordinates_text = extractor.get_coordinate_pairs(text)
    assert coordinates_text[0] == expected
