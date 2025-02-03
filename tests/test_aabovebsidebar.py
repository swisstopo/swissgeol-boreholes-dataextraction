"""Test suite for the find_depth_columns module."""

import fitz
from stratigraphy.sidebar import AAboveBSidebar
from stratigraphy.sidebar.a_above_b_sidebar import generate_alternatives
from stratigraphy.sidebar.sidebarentry import DepthColumnEntry


def test_aabovebsidebar_closetoarithmeticprogression():  # noqa: D103
    """Test the close_to_arithmetic_progression method of the AAboveBSidebar class."""
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=1),
            DepthColumnEntry(fitz.Rect(), value=2),
            DepthColumnEntry(fitz.Rect(), value=3),
            DepthColumnEntry(fitz.Rect(), value=4),
            DepthColumnEntry(fitz.Rect(), value=5),
        ]
    )
    assert sidebar.close_to_arithmetic_progression(), "The sidebar should be recognized as arithmetic progression"

    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=0.2),
            DepthColumnEntry(fitz.Rect(), value=0.3),
            DepthColumnEntry(fitz.Rect(), value=0.4),
        ]
    )
    assert sidebar.close_to_arithmetic_progression(), "The sidebar should be recognized as arithmetic progression"

    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=17.6),
            DepthColumnEntry(fitz.Rect(), value=18.15),
            DepthColumnEntry(fitz.Rect(), value=18.65),
            DepthColumnEntry(fitz.Rect(), value=19.3),
            DepthColumnEntry(fitz.Rect(), value=19.9),
            DepthColumnEntry(fitz.Rect(), value=20.5),
        ]
    )
    assert (
        not sidebar.close_to_arithmetic_progression()
    ), "The sidebar should not be recognized as arithmetic progression"


def test_aabovebsidebar_makeascending():  # noqa: D103
    """Test the make_ascending method of the AAboveBSidebar class."""

    def test(in_values, out_values):
        sidebar = AAboveBSidebar([DepthColumnEntry(fitz.Rect(), value=value) for value in in_values])
        result = [entry.value for entry in sidebar.make_ascending().entries]
        assert result == out_values, f"Expected {out_values}, but got {result}"

    # Basic transformation for values greater than the median, correct by factor 100
    test([1.0, 200.0, 3.0], [1.0, 2.0, 3.0])
    test([100.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    test([1.0, 2.0, 300.0], [1.0, 2.0, 3.0])

    # Basic transformation for values greater than the median, correct by factor 10
    test([1.0, 20.0, 300.0], [1.0, 20.0, 30.0])
    test([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0])
    test([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])

    ## Transforming OCR mistakes
    test([0.5, 4.0, 2.0, 5.0], [0.5, 1.0, 2.0, 5.0])
    test([4.0, 4.4, 4.4, 5.0], [4.0, 4.1, 4.4, 5.0])

    # ensure a "noise" value "0.0" does not influence the result
    test([1.0, 2.0, 3.0, 0.0, 4.0], [1.0, 2.0, 3.0, 0.0, 4.0])


def test_generate_alternatives():
    """Test generate_alternatives function for alternative options to OCR mistakes."""
    assert generate_alternatives(4) == [4, 1]
    assert generate_alternatives(14) == [14, 11]
    assert generate_alternatives(441) == [441, 411, 141, 111]
    assert generate_alternatives(123) == [123]
    assert generate_alternatives(4.4) == [4.4, 4.1, 1.4, 1.1]


def test_valid_value():
    """Test _valid_value helper function for make_ascending method of the AAboveBSidebar class."""
    entries = [DepthColumnEntry(None, 1), DepthColumnEntry(None, 2), DepthColumnEntry(None, 3)]
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
            DepthColumnEntry(fitz.Rect(), value=1),
            DepthColumnEntry(fitz.Rect(), value=2),
            DepthColumnEntry(fitz.Rect(), value=3),
            DepthColumnEntry(fitz.Rect(), value=4),
            DepthColumnEntry(fitz.Rect(), value=5),
        ]
    )
    assert sidebar.is_strictly_increasing(), "The sidebar should be strictly increasing"

    # Case 2: Not strictly increasing (equal values)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=1),
            DepthColumnEntry(fitz.Rect(), value=2),
            DepthColumnEntry(fitz.Rect(), value=2),
            DepthColumnEntry(fitz.Rect(), value=4),
        ]
    )
    assert not sidebar.is_strictly_increasing(), "The sidebar should not be strictly increasing"

    # Case 3: Not strictly increasing (decreasing)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=5),
            DepthColumnEntry(fitz.Rect(), value=4),
            DepthColumnEntry(fitz.Rect(), value=3),
            DepthColumnEntry(fitz.Rect(), value=2),
        ]
    )
    assert not sidebar.is_strictly_increasing(), "The sidebar should not be strictly increasing"

    # Case 4: Single entry (trivial)
    sidebar = AAboveBSidebar(
        [
            DepthColumnEntry(fitz.Rect(), value=1),
        ]
    )
    assert sidebar.is_strictly_increasing(), "A single entry should be considered strictly increasing"

    # Case 5: Empty
    sidebar = AAboveBSidebar([])
    assert sidebar.is_strictly_increasing(), "An empty list should be considered strictly increasing"
