"""Test suite for the coordinate_extraction module."""

import fitz
import pytest
from stratigraphy import DATAPATH
from stratigraphy.coordinates.coordinate_extraction import (
    Coordinate,
    CoordinateEntry,
    CoordinateExtractor,
    LV03Coordinate,
    LV95Coordinate,
)
from stratigraphy.util.line import TextLine, TextWord


def test_strLV95():  # noqa: D103
    """Test the string representation of an LV95Coordinate object."""
    coord = LV95Coordinate(
        east=CoordinateEntry(coordinate_value=2789456),
        north=CoordinateEntry(coordinate_value=1123012),
        rect=fitz.Rect(),
        page=1,
    )
    assert str(coord) == "E: 2'789'456, N: 1'123'012"


def test_to_jsonLV95():  # noqa: D103
    """Test the to_json method of an LV95Coordinate object."""
    coord = LV95Coordinate(
        east=CoordinateEntry(coordinate_value=2789456),
        north=CoordinateEntry(coordinate_value=1123012),
        rect=fitz.Rect(0, 1, 2, 3),
        page=1,
    )
    assert coord.to_json() == {"E": 2789456, "N": 1123012, "rect": [0, 1, 2, 3], "page": 1}


def test_swap_coordinates():  # noqa: D103
    """Test the swapping of coordinates in an LV95Coordinate object."""
    north = CoordinateEntry(coordinate_value=789456)
    east = CoordinateEntry(coordinate_value=123012)
    coord = LV95Coordinate(north=north, east=east, rect=fitz.Rect(), page=1)
    assert coord.east == north
    assert coord.north == east


def test_strLV03():  # noqa: D103
    """Test the string representation of an LV03Coordinate object."""
    coord = LV03Coordinate(
        east=CoordinateEntry(coordinate_value=789456),
        north=CoordinateEntry(coordinate_value=123012),
        rect=fitz.Rect(),
        page=1,
    )
    assert str(coord) == "E: 789'456, N: 123'012"


def test_to_jsonLV03():  # noqa: D103
    """Test the to_json method of an LV03Coordinate object."""
    coord = LV03Coordinate(
        east=CoordinateEntry(coordinate_value=789456),
        north=CoordinateEntry(coordinate_value=123012),
        rect=fitz.Rect(0, 1, 2, 3),
        page=1,
    )
    assert coord.to_json() == {"E": 789456, "N": 123012, "rect": [0, 1, 2, 3], "page": 1}


doc = fitz.open(DATAPATH.parent / "example" / "example_borehole_profile.pdf")
extractor = CoordinateExtractor(doc)


def test_CoordinateExtractor_extract_coordinates():  # noqa: D103
    """Test the extraction of coordinates from a PDF document."""
    # Assuming there is a method called 'extract' in CoordinateExtractor class
    coordinates = extractor.extract_data()
    # Check if the returned value is a list
    assert isinstance(coordinates, Coordinate)
    assert repr(coordinates.east) == "615'790"
    assert repr(coordinates.north) == "157'500"


def _create_simple_lines(text_lines: list[str]) -> list[TextLine]:
    """Create a list of TextLine objects from a list of text lines."""
    page_number = 1
    return [
        TextLine(
            [
                TextWord(fitz.Rect(word_index, line_index, word_index + 1, line_index + 1), word_text, page_number)
                for word_index, word_text in enumerate(text_line.split(" "))
            ]
        )
        for line_index, text_line in enumerate(text_lines)
    ]


def test_CoordinateExtractor_find_coordinate_key():  # noqa: D103
    """Test the extraction of the coordinate key from a list of text lines."""
    lines = _create_simple_lines(
        ["This is a sample text", "followed by a key with a spelling mistake", "Ko0rdinate 615.790 / 157.500"]
    )
    key_line = extractor.find_feature_key(lines)
    assert key_line[0].text == "Ko0rdinate 615.790 / 157.500"

    lines = _create_simple_lines(["An all-uppercase key should be matched", "COORDONNEES"])
    key_line = extractor.find_feature_key(lines)
    assert key_line[0].text == "COORDONNEES"

    lines = _create_simple_lines(["This is a sample text", "without any relevant key"])
    key_line = extractor.find_feature_key(lines)
    assert not key_line


def test_CoordinateExtractor_get_coordinates_with_x_y_labels():  # noqa: D103
    """Test the extraction of coordinates with explicit "X" and "Y" labels."""
    lines = _create_simple_lines(
        [
            "X = 2 600 000",
            "x = 1'200'001",
            "X = 2 600 002",
            "X = 2 600 003",
            "some noise",
            "Y = 1'200'000",
            "y = 2 600 001",
            "Y = 1'999'999",
        ]
    )
    coordinates = extractor.get_coordinates_with_x_y_labels(lines, page=1)

    # coordinates with explicit "X" and "Y" labels are found, even when they are further apart
    assert coordinates[0].east.coordinate_value == 2600000
    assert coordinates[0].north.coordinate_value == 1200000
    # 1st X-value is only combined with the 1st Y-value, 2nd X-value with 2nd Y-value, etc.
    # Values are swapped when necessary
    assert coordinates[1].east.coordinate_value == 2600001
    assert coordinates[1].north.coordinate_value == 1200001
    # ignore invalid coordinates and additional values that are only available with "X" or "Y" label, but not both
    assert len(coordinates) == 2


def test_CoordinateExtractor_get_coordinates_near_key():  # noqa: D103
    """Test the extraction of coordinates near a key."""
    lines = _create_simple_lines(
        [
            "This is a sample text followed by a key with a spelling",
            "mistake Ko0rdinate and some noise 615.79o / 157’ 500 in the middle.",
            "and a line immediately below 600 001 / 200 001",
            "and more lines below",
            "and more lines below",
            "and more lines below",
            "and something far below 600 002 / 200 002",
        ]
    )
    coordinates = extractor.get_feature_near_key(lines, page=1, page_width=100)

    # coordinates on the same line as the key are found, and OCR errors are corrected
    assert coordinates[0].east.coordinate_value == 615790
    assert coordinates[0].north.coordinate_value == 157500
    # coordinates immediately below is also found
    assert coordinates[1].east.coordinate_value == 600001
    assert coordinates[1].north.coordinate_value == 200001
    # no coordinates are found far down from the coordinates key
    assert len(coordinates) == 2


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
def test_CoordinateExtractor_get_coordinates_from_lines(text, expected):  # noqa: D103
    """Test the extraction of coordinates from a list of text lines."""
    lines = _create_simple_lines([text])
    coordinates = extractor.get_coordinates_from_lines(lines, page=1)
    expected_east, expected_north = expected
    assert coordinates[0].east.coordinate_value == expected_east
    assert coordinates[0].north.coordinate_value == expected_north
    assert coordinates[0].page == 1


def test_CoordinateExtractor_get_coordinates_from_lines_rect():  # noqa: D103
    """Test the extraction of coordinates from a list of text lines with different rect formats."""
    lines = _create_simple_lines(["start", "2600000 1200000", "end"])
    coordinates = extractor.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].rect == lines[1].rect
    assert coordinates[0].page == 1

    lines = _create_simple_lines(["start", "2600000", "1200000", "end"])
    coordinates = extractor.get_coordinates_from_lines(lines, page=1)
    expected_rect = lines[1].rect
    expected_rect.include_rect(lines[2].rect)
    assert coordinates[0].rect == expected_rect
    assert coordinates[0].page == 1
