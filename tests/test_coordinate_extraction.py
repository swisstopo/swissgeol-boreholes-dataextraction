"""Test suite for the coordinate_extraction module."""

import pymupdf
import pytest

from extraction import DATAPATH
from extraction.features.metadata.coordinate_extraction import (
    Coordinate,
    CoordinateEntry,
    CoordinateExtractor,
    LV03Coordinate,
    LV95Coordinate,
)
from swissgeol_doc_processing.text.textline import TextLine, TextWord


def test_strLV95():  # noqa: D103
    """Test the string representation of an LV95Coordinate object."""
    coord = LV95Coordinate(
        east=CoordinateEntry(coordinate_value=2789456), north=CoordinateEntry(coordinate_value=1123012)
    )
    assert str(coord) == "E: 2789456, N: 1123012"


def test_to_jsonLV95():  # noqa: D103
    """Test the to_json method of an LV95Coordinate object."""
    coord = LV95Coordinate(
        east=CoordinateEntry(coordinate_value=2789456), north=CoordinateEntry(coordinate_value=1123012)
    )
    assert coord.to_json() == {"E": 2789456, "N": 1123012}


def test_swap_coordinates():  # noqa: D103
    """Test the swapping of coordinates in an LV95Coordinate object."""
    north = CoordinateEntry(coordinate_value=789456)
    east = CoordinateEntry(coordinate_value=123012)
    coord = LV95Coordinate(north=north, east=east)
    assert coord.east == north
    assert coord.north == east


def test_strLV03():  # noqa: D103
    """Test the string representation of an LV03Coordinate object."""
    coord = LV03Coordinate(
        east=CoordinateEntry(coordinate_value=789456), north=CoordinateEntry(coordinate_value=123012)
    )
    assert str(coord) == "E: 789456, N: 123012"


def test_to_jsonLV03():  # noqa: D103
    """Test the to_json method of an LV03Coordinate object."""
    coord = LV03Coordinate(
        east=CoordinateEntry(coordinate_value=789456), north=CoordinateEntry(coordinate_value=123012)
    )
    assert coord.to_json() == {"E": 789456, "N": 123012}


doc = pymupdf.open(DATAPATH.parent / "example" / "example_borehole_profile.pdf")
doc_with_digits_in_coordinates = pymupdf.open(DATAPATH.parent / "example" / "A7367.pdf")
extractor_de = CoordinateExtractor("de")
extractor_fr = CoordinateExtractor("fr")


def test_CoordinateExtractor_extract_coordinates():  # noqa: D103
    """Test the extraction of coordinates from a PDF document."""
    # Assuming there is a method called 'extract' in CoordinateExtractor class
    coordinates = extractor_de.extract_coordinates(doc)[0]
    # Check if the returned value is a list
    assert isinstance(coordinates.feature, Coordinate)
    assert repr(coordinates.feature.east) == "615'790.0"
    assert repr(coordinates.feature.north) == "157'500.0"


def test_CoordinateExtractor_extract_coordinates_with_digits_in_coordinates():  # noqa: D103
    """Test the extraction of coordinates from a PDF document with digits in the coordinates."""
    # Assuming there is a method called 'extract' in CoordinateExtractor class
    coordinates = CoordinateExtractor("de").extract_coordinates(doc_with_digits_in_coordinates)[0]
    # Check if the returned value is a list
    assert isinstance(coordinates.feature, Coordinate)
    assert repr(coordinates.feature.east) == "607'562.0"
    assert repr(coordinates.feature.north) == "187'087.5"


def _create_simple_lines(text_lines: list[str]) -> list[TextLine]:
    """Create a list of TextLine objects from a list of text lines."""
    page_number = 1
    return [
        TextLine(
            [
                TextWord(pymupdf.Rect(word_index, line_index, word_index + 1, line_index + 1), word_text, page_number)
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
    key_line = extractor_de.find_feature_key(lines)
    assert key_line[0].text == "Ko0rdinate 615.790 / 157.500"

    lines = _create_simple_lines(["An all-uppercase key should be matched", "COORDONNEES"])
    key_line = extractor_fr.find_feature_key(lines)
    assert key_line[0].text == "COORDONNEES"

    lines = _create_simple_lines(["This is a sample text", "without any relevant key"])
    key_line = extractor_de.find_feature_key(lines)
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
    coordinates = extractor_de.get_coordinates_with_x_y_labels(lines, page=1)

    # coordinates with explicit "X" and "Y" labels are found, even when they are further apart
    assert coordinates[0].feature.east.coordinate_value == 2600000
    assert coordinates[0].feature.north.coordinate_value == 1200000
    # 1st X-value is only combined with the 1st Y-value, 2nd X-value with 2nd Y-value, etc.
    # Values are swapped when necessary
    assert coordinates[1].feature.east.coordinate_value == 2600001
    assert coordinates[1].feature.north.coordinate_value == 1200001
    # ignore invalid coordinates and additional values that are only available with "X" or "Y" label, but not both
    assert len(coordinates) == 2


def test_get_axis_aligned_lines():
    """Test the extraction of lines near feature key."""
    rect_key = pymupdf.Rect(x0=200, y0=200, x1=300, y1=250)

    # Key line
    key_line = TextLine([TextWord(pymupdf.Rect(200, 200, 300, 400), "Linie1", page=1)])
    # Inside horizontal range (right)
    inside_right = TextLine([TextWord(pymupdf.Rect(310, 200, 410, 250), "Linie2", page=1)])
    # Inside vertical range (below)
    inside_below = TextLine([TextWord(pymupdf.Rect(200, 260, 300, 310), "Linie3", page=1)])
    # Outside vertical and horizontal range (above)
    outside_above = TextLine([TextWord(pymupdf.Rect(200, 140, 300, 190), "Linie4", page=1)])
    # Completely outside both ranges (diagonal)
    outside_diagonal = TextLine([TextWord(pymupdf.Rect(310, 260, 410, 310), "Linie5", page=1)])
    # Edge case: exactly on horizontal limit
    boundary_left = TextLine([TextWord(pymupdf.Rect(100, 200, 200, 250), "Linie6", page=1)])
    # Edge case: exactly on edge of horizontal limit and vertical limit
    boundary_edge_above = TextLine([TextWord(pymupdf.Rect(300, 250, 400, 300), "Linie7", page=1)])
    # # Inside vertical limit, overlap with horizontal limit
    overlap_right = TextLine([TextWord(pymupdf.Rect(250, 200, 350, 250), "Linie8", page=1)])
    # # Overlap with vertical and horizontal limit
    overlap_right_below = TextLine([TextWord(pymupdf.Rect(250, 225, 350, 275), "Linie9", page=1)])

    text_lines = [
        key_line,
        inside_right,
        inside_below,
        outside_above,
        outside_diagonal,
        boundary_left,
        boundary_edge_above,
        overlap_right,
        overlap_right_below,
    ]

    # CoordinateExtractor searches (x1-x0) *10 to right and (y1-y0) * 3 below rect_key
    # Thus, x1 limit = 1300, y0 limit = 150
    feature_lines = extractor_de.get_axis_aligned_lines(lines=text_lines, rect=rect_key)
    expected_lines = [key_line, overlap_right, inside_below, inside_right, overlap_right_below]

    for feature_line in feature_lines:
        assert feature_line in expected_lines, f"Unexpected feature line: {feature_line}"

    for expected_line in expected_lines:
        assert expected_line in feature_lines, f"Expected line is missing: {expected_line}"


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
    coordinates = extractor_de.get_coordinates_near_key(lines, page=1)

    # coordinates on the same line as the key are found, and OCR errors are corrected
    assert coordinates[0].feature.east.coordinate_value == 615790
    assert coordinates[0].feature.north.coordinate_value == 157500
    # coordinates immediately below is also found
    assert coordinates[1].feature.east.coordinate_value == 600001
    assert coordinates[1].feature.north.coordinate_value == 200001
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
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    expected_east, expected_north = expected
    assert coordinates[0].feature.east.coordinate_value == expected_east
    assert coordinates[0].feature.north.coordinate_value == expected_north
    assert coordinates[0].page_number == 1


def test_CoordinateExtractor_get_coordinates_from_lines_rect():  # noqa: D103
    """Test the extraction of coordinates from a list of text lines with different rect formats."""
    lines = _create_simple_lines(["start", "2600000 1200000", "end"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].rect == lines[1].rect
    assert coordinates[0].page_number == 1

    lines = _create_simple_lines(["start", "2600000", "1200000", "end"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    expected_rect = lines[1].rect
    expected_rect.include_rect(lines[2].rect)
    assert coordinates[0].rect == expected_rect
    assert coordinates[0].page_number == 1

    # Example from 269126143-bp.pdf (a slash in the middle of the coordinates as misread by OCR as the digit 1)
    lines = _create_simple_lines(["269578211260032"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].feature.east.coordinate_value == 2695782
    assert coordinates[0].feature.north.coordinate_value == 1260032


def test_get_single_decimal_coordinates():
    """Test the extraction of decimal coordinates from a list of text lines."""
    lines = _create_simple_lines(["615.790.6 / 157.500.5"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].feature.east.coordinate_value == 615790.6
    assert coordinates[0].feature.north.coordinate_value == 157500.5

    lines = _create_simple_lines(["2600000.6 / 1200000.5"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].feature.east.coordinate_value == 2600000.6
    assert coordinates[0].feature.north.coordinate_value == 1200000.5


def test_get_double_decimal_coordinates():
    """Test the extraction of decimal coordinates from a list of text lines."""
    lines = _create_simple_lines(["615.790.64 / 157.500.55"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].feature.east.coordinate_value == 615790.64
    assert coordinates[0].feature.north.coordinate_value == 157500.55

    lines = _create_simple_lines(["2600000.64 / 1200000.55"])
    coordinates = extractor_de.get_coordinates_from_lines(lines, page=1)
    assert coordinates[0].feature.east.coordinate_value == 2600000.64
    assert coordinates[0].feature.north.coordinate_value == 1200000.55
