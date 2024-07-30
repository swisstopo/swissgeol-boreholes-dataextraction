"""Test suite for the find_depth_columns module."""

import fitz
import pytest
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.find_depth_columns import depth_column_entries, find_depth_columns, find_layer_depth_columns
from stratigraphy.util.line import TextLine, TextWord


def test_depth_column_entries():  # noqa: D103
    """Test the depth_column_entries function."""
    page_number = 1
    all_words = [
        TextWord(fitz.Rect(0, 0, 5, 1), "10.00m", page_number),
        TextWord(fitz.Rect(0, 2, 5, 3), "20.0m", page_number),
        TextWord(fitz.Rect(0, 4, 5, 5), "30.0m", page_number),
        TextWord(fitz.Rect(0, 6, 5, 7), "40.0m", page_number),
    ]
    entries = depth_column_entries(all_words, page_number, include_splits=False)
    assert len(entries) == 4, "There should be 4 entries"
    assert pytest.approx(entries[0].value) == 10.0, "The first entry should have a value of 10.0"
    assert pytest.approx(entries[1].value) == 20.0, "The second entry should have a value of 20.0"
    assert pytest.approx(entries[2].value) == 30.0, "The third entry should have a value of 30.0"
    assert pytest.approx(entries[3].value) == 40.0, "The fourth entry should have a value of 40.0"


def test_depth_column_entries_with_splits():  # noqa: D103
    """Test the depth_column_entries function with include_splits=True."""
    page_number = 1
    all_words = [
        TextLine([TextWord(fitz.Rect(0, 0, 10, 1), "10.00-20.0m", page_number)]),
        TextLine([TextWord(fitz.Rect(0, 2, 10, 3), "30.0-40.0m", page_number)]),
    ]
    entries = depth_column_entries(all_words, page_number, include_splits=True)
    assert len(entries) == 4, "There should be 4 entries"
    assert entries[0].value == 10.0, "The first entry should have a value of 10.0"
    assert entries[1].value == 20.0, "The second entry should have a value of 20.0"
    assert entries[2].value == 30.0, "The third entry should have a value of 30.0"
    assert entries[3].value == 40.0, "The fourth entry should have a value of 40.0"


def test_depth_column_entries_with_leading_character():  # noqa: D103
    """Test the depth_column_entries function with a leading character."""
    page_number = 1
    all_words = [
        TextWord(fitz.Rect(0, 0, 5, 1), "0.00m", page_number),
        TextWord(fitz.Rect(0, 2, 5, 3), ".2m", page_number),  # this is a test for an ocr error from '-2m' to '.2m'
        TextWord(fitz.Rect(0, 4, 5, 5), "-3.0m", page_number),
        TextWord(fitz.Rect(0, 6, 5, 7), ".4.2m", page_number),
    ]
    entries = depth_column_entries(all_words, page_number, include_splits=True)
    assert len(entries) == 4, "There should be 4 entries"
    assert entries[0].value == 0.0, "The first entry should have a value of 0"
    assert entries[1].value == 2.0, "The second entry should have a value of 2.0"
    assert entries[2].value == 3.0, "The third entry should have a value of 3.0"
    assert entries[3].value == 4.2, "The fourth entry should have a value of 4.2"


all_words_find_depth_column = [
    TextWord(fitz.Rect(0, 0, 5, 1), "10.00m", 1),
    TextWord(fitz.Rect(20, 0, 30, 1), "Kies, Torf und Sand", 1),
    TextWord(fitz.Rect(20, 2, 30, 3), "Kies, verwittert.", 1),
    TextWord(fitz.Rect(0, 2, 5, 3), "20.0m", 1),
    TextWord(fitz.Rect(0, 4, 5, 5), "30.0m", 1),
    TextWord(fitz.Rect(0, 6, 5, 7), "40.0m", 1),
    TextWord(fitz.Rect(0, 8, 5, 9), "50.0m", 1),
]


def test_find_depth_columns_arithmetic_progression():  # noqa: D103
    """Test the find_depth_columns function with an arithmetic progression."""
    entries = [
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 10.0, 1),
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 20.0, 1),
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 30.0, 1),
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 40.0, 1),
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 50.0, 1),
    ]
    page_number = 1

    columns = find_depth_columns(
        entries,
        all_words_find_depth_column,
        page_number,
        depth_column_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    assert len(columns) == 0, "There should be 0 columns as the above is a perfect arithmetic progression"


def test_find_depth_columns():  # noqa: D103
    """Test the find_depth_columns function."""
    page_number = 1
    entries = [
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 12.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 50.0, page_number),
    ]

    columns = find_depth_columns(
        entries,
        all_words_find_depth_column,
        page_number,
        depth_column_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    assert len(columns) == 1, "There should be 1 column"
    assert len(columns[0].entries) == 5, "The column should have 5 entries"
    assert pytest.approx(columns[0].entries[0].value) == 12.0, "The first entry should have a value of 12.0"
    assert pytest.approx(columns[0].entries[1].value) == 20.0, "The second entry should have a value of 20.0"
    assert pytest.approx(columns[0].entries[2].value) == 34.0, "The third entry should have a value of 34.0"
    assert pytest.approx(columns[0].entries[3].value) == 40.0, "The fourth entry should have a value of 40.0"
    assert pytest.approx(columns[0].entries[4].value) == 50.0, "The fourth entry should have a value of 50.0"


def test_two_columns_find_depth_columns():  # noqa: D103
    """Test the find_depth_columns function with two columns."""
    page_number = 1
    entries = [  # first depth column
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 12.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 50.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 0, 25, 1), 12.0, page_number),  # second depth column
        DepthColumnEntry(fitz.Rect(20, 2, 25, 3), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 4, 25, 5), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 6, 25, 7), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 8, 25, 9), 50.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 10, 25, 11), 61.0, page_number),
    ]
    columns = find_depth_columns(
        entries,
        all_words_find_depth_column,
        page_number,
        depth_column_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    assert len(columns) == 2, "There should be 2 columns"
    assert len(columns[0].entries) == 5, "The first column should have 5 entries"
    assert len(columns[1].entries) == 6, "The second column should have 6 entries"


all_words_find_layer_depth_column = [
    TextWord(fitz.Rect(0, 0, 5, 1), "12.00-20.0m", 1),
    TextWord(fitz.Rect(20, 0, 30, 1), "Kies, Torf und Sand", 1),
    TextWord(fitz.Rect(20, 2, 30, 3), "Kies, verwittert.", 1),
    TextWord(fitz.Rect(0, 2, 5, 3), "20.0-34.0m", 1),
    TextWord(fitz.Rect(0, 4, 5, 5), "34.0 - 40.0m", 1),
    TextWord(fitz.Rect(0, 6, 5, 7), "40.0-50m", 1),
    TextWord(fitz.Rect(0, 8, 5, 9), "50.0-60m", 1),
]


def test_find_layer_depth_columns():  # noqa: D103
    """Test the find_layer_depth_columns function."""
    page_number = 1
    entries = [
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 12.0, page_number),  # layer 12.0-20.0m
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 20.0, page_number),  # layer 20.0-34.0m
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 34.0, page_number),  # layer 34.0-40.0m
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 40.0, page_number),  # layer 40.0-50.0m
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 50.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 50.0, page_number),  # layer 50.0-60.0m
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 60.0, page_number),
    ]

    columns = find_layer_depth_columns(entries, all_words_find_depth_column)
    assert len(columns) == 1, "There should be 1 column"
    assert len(columns[0].entries) == 5, "The column should have 5 entries"
    assert columns[0].entries[0].start.value == 12.0, "The first entry should have a value of 12.0"
    assert columns[0].entries[0].end.value == 20.0, "The first entry should have a value of 20.0"
    assert columns[0].entries[1].start.value == 20.0, "The second entry should have a value of 20.0"
    assert columns[0].entries[1].end.value == 34.0, "The second entry should have a value of 34.0"
    assert columns[0].entries[2].start.value == 34.0, "The third entry should have a value of 34.0"
    assert columns[0].entries[2].end.value == 40.0, "The third entry should have a value of 40.0"
    assert columns[0].entries[3].start.value == 40.0, "The fourth entry should have a value of 40.0"
    assert columns[0].entries[3].end.value == 50.0, "The fourth entry should have a value of 50.0"
    assert columns[0].entries[4].start.value == 50.0, "The fourth entry should have a value of 50.0"
    assert columns[0].entries[4].end.value == 60.0, "The fourth entry should have a value of 60.0"


def test_two_columns_find_layer_depth_columns():  # noqa: D103
    """Test the find_layer_depth_columns function with two columns."""
    page_number = 1
    entries = [  # first depth column
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 12.0, page_number),  # layer 12.0-20.0m
        DepthColumnEntry(fitz.Rect(0, 0, 5, 1), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 20.0, page_number),  # layer 20.0-34.0m
        DepthColumnEntry(fitz.Rect(0, 2, 5, 3), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 34.0, page_number),  # layer 34.0-40.0m
        DepthColumnEntry(fitz.Rect(0, 4, 5, 5), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 40.0, page_number),  # layer 40.0-50.0m
        DepthColumnEntry(fitz.Rect(0, 6, 5, 7), 50.0, page_number),
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 50.0, page_number),  # layer 50.0-60.0m
        DepthColumnEntry(fitz.Rect(0, 8, 5, 9), 60.0, page_number),
        # second depth column
        DepthColumnEntry(fitz.Rect(20, 0, 25, 1), 12.0, page_number),  # layer 12.0-20.0m
        DepthColumnEntry(fitz.Rect(20, 0, 25, 1), 20.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 2, 25, 3), 20.0, page_number),  # layer 20.0-34.0m
        DepthColumnEntry(fitz.Rect(20, 2, 25, 3), 34.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 4, 25, 5), 34.0, page_number),  # layer 34.0-40.0m
        DepthColumnEntry(fitz.Rect(20, 4, 25, 5), 40.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 6, 25, 7), 40.0, page_number),  # layer 40.0-50.0m
        DepthColumnEntry(fitz.Rect(20, 6, 25, 7), 50.0, page_number),
        DepthColumnEntry(fitz.Rect(20, 8, 25, 9), 50.0, page_number),  # layer 50.0-60.0m
        DepthColumnEntry(fitz.Rect(20, 8, 25, 9), 60.0, page_number),
    ]
    columns = find_layer_depth_columns(entries, all_words_find_layer_depth_column)
    assert len(columns) == 2, "There should be 2 columns"
    assert len(columns[0].entries) == 5, "The first column should have 5 entries"
    assert len(columns[1].entries) == 5, "The second column should have 5 entries"
