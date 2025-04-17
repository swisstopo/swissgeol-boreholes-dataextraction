"""Test suite for the find_depth_columns module."""

import pymupdf
import pytest
import rtree
from borehole_extraction.extraction.stratigraphy.sidebar_extractor.a_above_b_sidebar_extractor import (
    AAboveBSidebarExtractor,
)
from borehole_extraction.extraction.stratigraphy.sidebar_extractor.a_to_b_sidebar_extractor import (
    AToBSidebarExtractor,
)
from borehole_extraction.extraction.stratigraphy.sidebar_extractor.depth_column_entry_extractor import (
    DepthColumnEntryExtractor,
)
from borehole_extraction.extraction.util_extraction.text.textline import TextWord

PAGE_NUMBER = 1


def test_depth_column_entries():  # noqa: D103
    """Test the DepthColumnEntryExtractor.find_in_words function."""
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 5, 1), "10.00m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "30.0m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "60.", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 10, 5, 11), "-70m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 12, 5, 13), "word.", PAGE_NUMBER),
    ]
    entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
    assert len(entries) == 7, "There should be 7 entries"
    assert pytest.approx(entries[0].value) == 10.0, "The first entry should have a value of 10.0"
    assert entries[0].has_decimal_point, "The first entry has a decimal point"

    assert pytest.approx(entries[1].value) == 20.0, "The second entry should have a value of 20.0"
    assert entries[1].has_decimal_point, "The second entry has a decimal point"

    assert pytest.approx(entries[2].value) == 30.0, "The third entry should have a value of 30.0"
    assert entries[2].has_decimal_point, "The third entry has a decimal point"

    assert pytest.approx(entries[3].value) == 40.0, "The fourth entry should have a value of 40.0"
    assert entries[3].has_decimal_point, "The forth entry has a decimal point"

    assert pytest.approx(entries[4].value) == 50.0, "The fifth entry should have a value of 50.0"
    assert not entries[4].has_decimal_point, "The fifth entry doesn't have a decimal point"

    assert pytest.approx(entries[5].value) == 60.0, "The sixth entry should have a value of 60.0"
    assert entries[5].has_decimal_point, "The sixth entry has a decimal point"

    assert pytest.approx(entries[6].value) == 70.0, "The seventh entry should have a value of 70.0"
    assert not entries[6].has_decimal_point, "The seventh entry doesn't have a decimal point"


def test_depth_column_entries_with_splits():  # noqa: D103
    """Test the DepthColumnEntry.find_in_words function with include_splits=True."""
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 10, 1), "10.00-20.0m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 10, 3), "30.0-40.0m", PAGE_NUMBER),
    ]
    entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=True)
    assert len(entries) == 4, "There should be 4 entries"
    assert entries[0].value == 10.0, "The first entry should have a value of 10.0"
    assert entries[1].value == 20.0, "The second entry should have a value of 20.0"
    assert entries[2].value == 30.0, "The third entry should have a value of 30.0"
    assert entries[3].value == 40.0, "The fourth entry should have a value of 40.0"


def test_depth_column_entries_with_leading_character():  # noqa: D103
    """Test the depth_column_entries function with a leading character."""
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 5, 1), "0.00m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), ".2m", PAGE_NUMBER),  # this is a test for an ocr error from '-2m' to '.2m'
        TextWord(pymupdf.Rect(0, 4, 5, 5), "-3.0m", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), ".4.2m", PAGE_NUMBER),
    ]
    entries = DepthColumnEntryExtractor.find_in_words(all_words, include_splits=True)
    assert len(entries) == 4, "There should be 4 entries"
    assert entries[0].value == 0.0, "The first entry should have a value of 0"
    assert entries[1].value == 2.0, "The second entry should have a value of 2.0"
    assert entries[2].value == 3.0, "The third entry should have a value of 3.0"
    assert entries[3].value == 4.2, "The fourth entry should have a value of 4.2"


def test_aabovebsidebarextractor_arithmetic_progression():  # noqa: D103
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 5, 1), "10.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "30.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50.0", PAGE_NUMBER),
    ]
    word_rtree = rtree.index.Index()
    for i, word in enumerate(all_words):
        word_rtree.insert(i, (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1))
    """Test the AAboveBSidebarExtractor with an arithmetic progression."""
    sidebars_noise = AAboveBSidebarExtractor.find_in_words(
        all_words,
        word_rtree,
        used_entry_rects=[],
        sidebar_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    columns = [sidebar_noise.sidebar for sidebar_noise in sidebars_noise]
    assert len(columns) == 0, "There should be 0 columns as the above is a perfect arithmetic progression"


def test_aabovebsidebarextractor():  # noqa: D103
    """Test the AAboveBSidebarExtractor."""
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 5, 1), "12.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50.0", PAGE_NUMBER),
    ]
    word_rtree = rtree.index.Index()
    for word in all_words:
        word_rtree.insert(id(word), (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1), obj=word)
    sidebars_noise = AAboveBSidebarExtractor.find_in_words(
        all_words,
        word_rtree,
        used_entry_rects=[],
        sidebar_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    columns = [sidebar_noise.sidebar for sidebar_noise in sidebars_noise]
    assert len(columns) == 1, "There should be 1 column"
    assert len(columns[0].entries) == 5, "The column should have 5 entries"
    assert pytest.approx(columns[0].entries[0].value) == 12.0, "The first entry should have a value of 12.0"
    assert pytest.approx(columns[0].entries[1].value) == 20.0, "The second entry should have a value of 20.0"
    assert pytest.approx(columns[0].entries[2].value) == 34.0, "The third entry should have a value of 34.0"
    assert pytest.approx(columns[0].entries[3].value) == 40.0, "The fourth entry should have a value of 40.0"
    assert pytest.approx(columns[0].entries[4].value) == 50.0, "The fourth entry should have a value of 50.0"


def test_aabovebsidebarextractor_two_column():  # noqa: D103
    """Test the AAboveBSidebarExtractor function with two columns."""
    all_words = [  # first depth column
        TextWord(pymupdf.Rect(0, 0, 5, 1), "12.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 0, 25, 1), "12.0", PAGE_NUMBER),  # second depth column
        TextWord(pymupdf.Rect(20, 2, 25, 3), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 4, 25, 5), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 6, 25, 7), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 8, 25, 9), "50.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 10, 25, 11), "61.0", PAGE_NUMBER),
    ]
    word_rtree = rtree.index.Index()
    for word in all_words:
        word_rtree.insert(id(word), (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1), obj=word)
    sidebars_noise = AAboveBSidebarExtractor.find_in_words(
        all_words,
        word_rtree,
        used_entry_rects=[],
        sidebar_params={"noise_count_threshold": 1.25, "noise_count_offset": 0},
    )
    columns = [sidebar_noise.sidebar for sidebar_noise in sidebars_noise]

    assert len(columns) == 2, "There should be 2 columns"
    assert {len(column.entries) for column in columns} == {5, 6}, "The columns should have 5 and 6 entries"


def test_atobsidebarextractor():  # noqa: D103
    """Test the AToBSidebarExtractor."""
    all_words = [
        TextWord(pymupdf.Rect(0, 0, 5, 1), "12.0", PAGE_NUMBER),  # layer 12.0-20.0m
        TextWord(pymupdf.Rect(0, 0, 5, 1), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0", PAGE_NUMBER),  # layer 20.0-34.0m
        TextWord(pymupdf.Rect(0, 2, 5, 3), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "34.0", PAGE_NUMBER),  # layer 34.0-40.0m
        TextWord(pymupdf.Rect(0, 4, 5, 5), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0", PAGE_NUMBER),  # layer 40.0-50.0m
        TextWord(pymupdf.Rect(0, 6, 5, 7), "50.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50.0", PAGE_NUMBER),  # layer 50.0-60.0m
        TextWord(pymupdf.Rect(0, 8, 5, 9), "60.0", PAGE_NUMBER),
    ]
    columns = AToBSidebarExtractor.find_in_words(all_words)
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


def test_atobsidebarextractor_two_columns():  # noqa: D103
    """Test the AToBSidebarExtractor with two columns."""
    all_words = [  # first depth column
        TextWord(pymupdf.Rect(0, 0, 5, 1), "12.0", PAGE_NUMBER),  # layer 12.0-20.0m
        TextWord(pymupdf.Rect(0, 0, 5, 1), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 2, 5, 3), "20.0", PAGE_NUMBER),  # layer 20.0-34.0m
        TextWord(pymupdf.Rect(0, 2, 5, 3), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 4, 5, 5), "34.0", PAGE_NUMBER),  # layer 34.0-40.0m
        TextWord(pymupdf.Rect(0, 4, 5, 5), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 6, 5, 7), "40.0", PAGE_NUMBER),  # layer 40.0-50.0m
        TextWord(pymupdf.Rect(0, 6, 5, 7), "50.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(0, 8, 5, 9), "50.0", PAGE_NUMBER),  # layer 50.0-60.0m
        TextWord(pymupdf.Rect(0, 8, 5, 9), "60.0", PAGE_NUMBER),
        # second depth column
        TextWord(pymupdf.Rect(20, 0, 25, 1), "12.0", PAGE_NUMBER),  # layer 12.0-20.0m
        TextWord(pymupdf.Rect(20, 0, 25, 1), "20.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 2, 25, 3), "20.0", PAGE_NUMBER),  # layer 20.0-34.0m
        TextWord(pymupdf.Rect(20, 2, 25, 3), "34.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 4, 25, 5), "34.0", PAGE_NUMBER),  # layer 34.0-40.0m
        TextWord(pymupdf.Rect(20, 4, 25, 5), "40.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 6, 25, 7), "40.0", PAGE_NUMBER),  # layer 40.0-50.0m
        TextWord(pymupdf.Rect(20, 6, 25, 7), "50.0", PAGE_NUMBER),
        TextWord(pymupdf.Rect(20, 8, 25, 9), "50.0", PAGE_NUMBER),  # layer 50.0-60.0m
        TextWord(pymupdf.Rect(20, 8, 25, 9), "60.0", PAGE_NUMBER),
    ]
    columns = AToBSidebarExtractor.find_in_words(all_words)
    assert len(columns) == 2, "There should be 2 columns"
    assert len(columns[0].entries) == 5, "The first column should have 5 entries"
    assert len(columns[1].entries) == 5, "The second column should have 5 entries"
