"""Test suite for the find_depth_columns module."""

import pymupdf
import pytest

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import (
    AAboveBSidebar,
    generate_alternatives,
)
from extraction.features.stratigraphy.sidebar.extractor.a_above_b_sidebar_extractor import AAboveBSidebarExtractor


@pytest.mark.parametrize(
    "input,expected",
    [
        (["1", "1.05", "2", "3", "4", "5.78", "6"], [1, 2, 3, 4]),
        (["10", "20", "30", "40", "50"], [10, 20, 30, 40, 50]),
        (["5", "10", "15", "20", "25", "30.5", "40.7"], [5, 10, 15, 20, 25]),
        (["3", "7", "12", "20"], []),
        (["10"], []),
        (["1.1", "2.2", "3.3", "4.4"], [1.1, 2.2, 3.3, 4.4]),
        (
            ["8", "16", "20", "24", "28", "52", "56", "60", "72", "88", "92", "100"],
            [],
        ),
        (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (["0.8", "2.4", "4.0"], []),
        (["0.2", "0.3", "0.4"], [0.2, 0.3, 0.4]),
        (["2", "4", "6", "8", "12"], [2, 4, 6, 8]),
        (
            [
                "0",
                "0.8",
                "1",
                "1.7",
                "2",
                "2.7",
                "3",
                "3.1",
                "3.5",
                "3.9",
                "4",
                "5",
                "6",
                "7",
                "7.4",
                "8",
                "8.5",
                "8.8",
                "9",
                "10",
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),  # inspired by 268124125-bp.pdf
    ],
)
def test_aabovebsidebar_arithmeticprogressionentries(input, expected):  # noqa: D103
    """Test the _arithmetic_progression_entries method of the AAboveBSidebarExtractor class."""
    entries = [
        DepthColumnEntry.from_string_value(pymupdf.Rect(), string_value=value, page_number=0) for value in input
    ]
    result = [entry.value for entry in AAboveBSidebarExtractor._arithmetic_progression_entries(entries)]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_aabovebsidebar_fixocrmistakes():  # noqa: D103
    """Test the fix_ocr_mistakes method of the AAboveBSidebar class."""

    def run_test(in_values, out_values):
        sidebar = AAboveBSidebar(
            [
                # TODO: actually specify the y-coordinate instead of using the index as a proxy
                DepthColumnEntry(rect=pymupdf.Rect(0, index, 0, index), value=value, page_number=0)
                for index, value in enumerate(in_values)
            ]
        )
        result = [entry.value for entry in sidebar.fix_ocr_mistakes().entries]
        assert result == out_values, f"Expected {out_values}, but got {result}"

    # Basic transformation for values greater than the median, correct by factor 100
    run_test([1.0, 200.0, 3.0], [1.0, 2.0, 3.0])
    run_test([100.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    run_test([1.0, 2.0, 300.0], [1.0, 2.0, 3.0])

    # Basic transformation for values greater than the median, correct by factor 10
    run_test([1.0, 20.0, 300.0], [1.0, 20.0, 30.0])
    run_test([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0])
    run_test([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])

    ## Transforming OCR mistakes
    run_test([0.5, 4.0, 2.0, 5.0], [0.5, 1.0, 2.0, 5.0])
    run_test([4.0, 4.4, 4.4, 4.7, 5.0], [4.0, 4.1, 4.4, 4.7, 5.0])

    # ensure a "noise" value "0.0" does not influence the result
    run_test([1.0, 2.0, 3.0, 0.0, 4.0], [1.0, 2.0, 3.0, 0.0, 4.0])

    # always preserve the inputs if they are already look good
    run_test([0.0, 0.1, 0.5, 6.0, 8.5, 10.0], [0.0, 0.1, 0.5, 6.0, 8.5, 10.0])

    # Test case for A11429
    run_test([558.4, 0.25, 230.0, 4.3, 12.04, 18268.0], [558.4, 0.25, 2.30, 4.3, 12.04, 182.68])

    # Test case for 267125358-bp.pdf (two boreholes in one column )
    run_test(
        [0.3, 1.0, 1.6, 1.9, 2.4, 3.2, 0.2, 0.4, 1.3, 2.3, 3.0],
        [0.3, 1.0, 1.6, 1.9, 2.4, 3.2, 0.2, 0.4, 1.3, 2.3, 3.0],
    )

    # edge case
    run_test([], [])


def test_generate_alternatives():
    """Test generate_alternatives function for alternative options to OCR mistakes."""
    assert generate_alternatives(4) == [4, 1]
    assert generate_alternatives(14) == [14, 11]
    assert generate_alternatives(441) == [441, 411, 141, 111]
    assert generate_alternatives(123) == [123]
    assert generate_alternatives(4.4) == [4.4, 4.1, 1.4, 1.1]


def test_aabovebsidebar_isstrictlyincreasing():  # noqa: D103
    """Test the is_strictly_increasing method of the AAboveBSidebar class."""
    # Case 1: Strictly increasing values
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=1, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=2, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=3, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=4, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=5, page_number=0),
        ]
    )
    assert sidebar.is_strictly_increasing(), "The sidebar should be strictly increasing"

    # Case 2: Not strictly increasing (equal values)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=1, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=2, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=2, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=4, page_number=0),
        ]
    )
    assert not sidebar.is_strictly_increasing(), "The sidebar should not be strictly increasing"

    # Case 3: Not strictly increasing (decreasing)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=5, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=4, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=3, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=2, page_number=0),
        ]
    )
    assert not sidebar.is_strictly_increasing(), "The sidebar should not be strictly increasing"

    # Case 4: Single entry (trivial)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=1, page_number=0),
        ]
    )
    assert sidebar.is_strictly_increasing(), "A single entry should be considered strictly increasing"

    # Case 5: Empty
    sidebar = AAboveBSidebar([])
    assert sidebar.is_strictly_increasing(), "An empty list should be considered strictly increasing"
