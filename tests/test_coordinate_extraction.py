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
from stratigraphy.util.line import TextLine, TextWord


def test_strLV95():  # noqa: D103
    coord = LV95Coordinate(CoordinateEntry(2789456), CoordinateEntry(1123012), fitz.Rect())
    assert str(coord) == "E: 2'789'456, N: 1'123'012"


def test_to_jsonLV95():  # noqa: D103
    coord = LV95Coordinate(CoordinateEntry(2789456), CoordinateEntry(1123012), fitz.Rect(0, 1, 2, 3))
    assert coord.to_json() == {"E": 2789456, "N": 1123012, "rect": [0, 1, 2, 3]}


def test_swap_coordinates():  # noqa: D103
    north = CoordinateEntry(789456)
    east = CoordinateEntry(123012)
    coord = LV95Coordinate(north=north, east=east, rect=fitz.Rect())
    assert coord.east == north
    assert coord.north == east


def test_strLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry(789456), CoordinateEntry(123012), rect=fitz.Rect())
    assert str(coord) == "E: 789'456, N: 123'012"


def test_to_jsonLV03():  # noqa: D103
    coord = LV03Coordinate(CoordinateEntry(789456), CoordinateEntry(123012), fitz.Rect(0, 1, 2, 3))
    assert coord.to_json() == {
        "E": 789456,
        "N": 123012,
        "rect": [0, 1, 2, 3],
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


def _create_simple_lines(text_lines: list[str]) -> list[TextLine]:
    return [
        TextLine(
            [
                TextWord(fitz.Rect(word_index, line_index, word_index + 1, line_index + 1), word_text)
                for word_index, word_text in enumerate(text_line.split(" "))
            ]
        )
        for line_index, text_line in enumerate(text_lines)
    ]


def test_CoordinateExtractor_find_coordinate_key():  # noqa: D103
    lines = _create_simple_lines(
        ["This is a sample text", "followed by a key with a spelling mistake", "Ko0rdinate 615.790 / 157.500"]
    )
    key_line = extractor.find_coordinate_key(lines)
    assert key_line.text == "Ko0rdinate 615.790 / 157.500"

    lines = _create_simple_lines(["An all-uppercase key should be matched", "COORDONNEES"])
    key_line = extractor.find_coordinate_key(lines)
    assert key_line.text == "COORDONNEES"

    lines = _create_simple_lines(["This is a sample text", "without any relevant key"])
    key_line = extractor.find_coordinate_key(lines)
    assert key_line is None


def test_CoordinateExtractor_get_coordinate_substring():  # noqa: D103
    lines = _create_simple_lines(
        [
            "This is a sample text followed by a key with a spelling",
            "mistake Ko0rdinate and some noise 615.79o / 157; 500 in the middle.",
            "and a line immediately below AAA",
            "and more lines below",
            "and more lines below",
            "and more lines below",
            "and something far below BBB",
        ]
    )
    substring = " ".join([line.text for line in extractor.get_coordinate_lines(lines, page_width=100)])
    assert "615.79o / 157; 500" in substring
    assert "AAA" in substring
    assert "BBB" not in substring


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
    lines = _create_simple_lines([text])
    coordinates = extractor.get_coordinates_from_lines(lines)
    expected_east, expected_north = expected
    assert coordinates[0].east.coordinate_value == expected_east
    assert coordinates[0].north.coordinate_value == expected_north
