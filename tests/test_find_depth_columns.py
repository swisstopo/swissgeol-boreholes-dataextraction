"""Test suite for the find_depth_columns module."""

import fitz
import pytest
from stratigraphy.util.find_depth_columns import depth_column_entries
from stratigraphy.util.line import DepthInterval, TextLine


def test_depth_column_entries():  # noqa: D103
    all_words = [
        TextLine([DepthInterval(fitz.Rect(0, 0, 5, 1), "10.00m")]),
        TextLine([DepthInterval(fitz.Rect(0, 2, 5, 3), "20.0m")]),
        TextLine([DepthInterval(fitz.Rect(0, 4, 5, 5), "30.0m")]),
        TextLine([DepthInterval(fitz.Rect(0, 6, 5, 7), "40.0m")]),
    ]
    entries = depth_column_entries(all_words, include_splits=False)
    assert len(entries) == 4, "There should be 4 entries"
    assert pytest.approx(entries[0].value) == 10.0, "The first entry should have a value of 10.0"
    assert pytest.approx(entries[1].value) == 20.0, "The second entry should have a value of 20.0"
    assert pytest.approx(entries[2].value) == 30.0, "The third entry should have a value of 30.0"
    assert pytest.approx(entries[3].value) == 40.0, "The fourth entry should have a value of 40.0"


def test_depth_column_entries_with_splits():  # noqa: D103
    all_words = [
        TextLine([DepthInterval(fitz.Rect(0, 0, 10, 1), "10.00-20.0m")]),
        TextLine([DepthInterval(fitz.Rect(0, 2, 10, 3), "30.0-40.0m")]),
    ]
    entries = depth_column_entries(all_words, include_splits=True)
    assert len(entries) == 4, "There should be 4 entries"
    assert entries[0].value == 10.0, "The first entry should have a value of 10.0"
    assert entries[1].value == 20.0, "The second entry should have a value of 20.0"
    assert entries[2].value == 30.0, "The third entry should have a value of 30.0"
    assert entries[3].value == 40.0, "The fourth entry should have a value of 40.0"
