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
    coord = LV95Coordinate(CoordinateEntry("789", "456"), CoordinateEntry("123", "012"))
    assert repr(coord) == "longitude: 2.789.456, latitude: 1.123.012"


def test_to_jsonLV95():  # noqa: D103
    coord = LV95Coordinate(CoordinateEntry("789", "456"), CoordinateEntry("123", "012"))
    assert coord.to_json() == {
        "longitude": "2.789.456",
        "latitude": "1.123.012",
    }


def test_swap_coordinates():  # noqa: D103
    latitude = CoordinateEntry("789", "456")
    longitude = CoordinateEntry("123", "012")
    coord = LV95Coordinate(latitude=latitude, longitude=longitude)
    assert coord.longitude == latitude
    assert coord.latitude == longitude


def test_reprLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry("789", "456"), CoordinateEntry("123", "012"))
    assert repr(coord) == "longitude: 789.456, latitude: 123.012"


def test_to_jsonLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry("789", "456"), CoordinateEntry("123", "012"))
    assert coord.to_json() == {
        "longitude": "789.456",
        "latitude": "123.012",
    }


@pytest.mark.parametrize("first_entry,second_entry", [("123", "456"), ("789", "012")])
def test_coordinate_entry(first_entry, second_entry):  # noqa: D103
    entry = CoordinateEntry(first_entry, second_entry)
    assert entry.first_entry == first_entry
    assert entry.second_entry == second_entry


doc = fitz.open(DATAPATH.parent / "example" / "example_borehole_profile.pdf")
extractor = CoordinateExtractor(doc)


def test_CoordinateExtractor_extract_coordinates():  # noqa: D103
    # Assuming there is a method called 'extract' in CoordinateExtractor class
    coordinates = extractor.extract_coordinates()
    # Check if the returned value is a list
    assert isinstance(coordinates, Coordinate)
    assert repr(coordinates.longitude) == "615.790"
    assert repr(coordinates.latitude) == "157.500"


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
            " 615.790 / 157.500",
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X= 615.790 / Y157.500 and some noise",
            "X= 615.790 / Y157.500",
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X: 2'615'790 / 1'157'500 and some noise",
            "X: 2'615'790 / 1'157'500",
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate X 2615790 / 1157500 and some noise",
            "X 2615790 / 1157500",
        ),
        (
            "sample text followed by a key with a spelling mistake Ko0rdinate 615790 / 157500 and some noise",
            " 615790 / 157500",
        ),
    ],
)
def test_CoordinateExtractor_get_coordinates_text(text, expected):  # noqa: D103
    coordinates_text = extractor.get_coordinates_text(text)
    assert coordinates_text[0] == expected
