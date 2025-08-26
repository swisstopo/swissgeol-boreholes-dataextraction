"""Test suite for the find_depth_columns module."""

import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import (
    AAboveBSidebar,
    generate_alternatives,
)


def test_aabovebsidebar_closetoarithmeticprogression():  # noqa: D103
    """Test the close_to_arithmetic_progression method of the AAboveBSidebar class."""
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=1, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=2, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=3, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=4, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=5, page_number=0),
        ]
    )
    assert sidebar.close_to_arithmetic_progression(), "The sidebar should be recognized as arithmetic progression"

    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=0.2, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=0.3, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=0.4, page_number=0),
        ]
    )
    assert sidebar.close_to_arithmetic_progression(), "The sidebar should be recognized as arithmetic progression"

    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=17.6, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=18.15, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=18.65, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=19.3, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=19.9, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=20.5, page_number=0),
        ]
    )
    assert not sidebar.close_to_arithmetic_progression(), (
        "The sidebar should not be recognized as arithmetic progression"
    )


def test_aabovebsidebar_removeintegerscale():  # noqa: D103
    """Test the remove_integer_scale method of the AAboveBSidebar class."""

    def run_test(in_values, out_values):
        sidebar = AAboveBSidebar(
            [
                DepthColumnEntry.from_string_value(pymupdf.Rect(), string_value=value, page_number=0)
                for value in in_values
            ]
        )
        result = [entry.value for entry in sidebar.remove_integer_scale().entries]
        assert result == out_values, f"Expected {out_values}, but got {result}"

    run_test(["1.05", "2", "3", "4", "5.78", "6"], [1.05, 5.78])
    run_test(["10", "20", "30", "40", "50"], [])
    run_test(["5", "10", "15", "20", "25", "30.5", "40.7"], [30.5, 40.7])
    run_test(["3", "7", "12", "20"], [3, 7, 12, 20])
    run_test(["10"], [10])
    run_test(["1.1", "2.2", "3.3", "4.4"], [1.1, 2.2, 3.3, 4.4])


def test_aabovebsidebar_makeascending():  # noqa: D103
    """Test the make_ascending method of the AAboveBSidebar class."""

    def run_test(in_values, out_values):
        sidebar = AAboveBSidebar(
            [DepthColumnEntry(rect=pymupdf.Rect(), value=value, page_number=0) for value in in_values]
        )
        result = [entry.value for entry in sidebar.make_ascending().entries]
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
    run_test([4.0, 4.4, 4.4, 5.0], [4.0, 4.1, 4.4, 5.0])

    # ensure a "noise" value "0.0" does not influence the result
    run_test([1.0, 2.0, 3.0, 0.0, 4.0], [1.0, 2.0, 3.0, 0.0, 4.0])

    # edge case
    run_test([], [])


def test_generate_alternatives():
    """Test generate_alternatives function for alternative options to OCR mistakes."""
    assert generate_alternatives(4) == [4, 1]
    assert generate_alternatives(14) == [14, 11]
    assert generate_alternatives(441) == [441, 411, 141, 111]
    assert generate_alternatives(123) == [123]
    assert generate_alternatives(4.4) == [4.4, 4.1, 1.4, 1.1]


def test_valid_value():
    """Test _valid_value helper function for make_ascending method of the AAboveBSidebar class."""
    entries = [
        DepthColumnEntry(rect=None, value=1, page_number=0),
        DepthColumnEntry(rect=None, value=2, page_number=0),
        DepthColumnEntry(rect=None, value=3, page_number=0),
    ]
    sidebar = AAboveBSidebar(entries)

    assert sidebar._valid_value(1, 2) is True
    assert sidebar._valid_value(1, 3) is False
    assert sidebar._valid_value(1, 1.5) is True
    assert sidebar._valid_value(0, 2) is False
    assert sidebar._valid_value(2, 3.5) is True


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
