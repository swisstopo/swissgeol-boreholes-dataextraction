"""Test suite for the find_depth_columns module."""

import fitz
from stratigraphy.depth.depthcolumnentry import DepthColumnEntry
from stratigraphy.sidebar import AAboveBSidebar


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
        assert [
            entry.value for entry in sidebar.make_ascending().entries
        ] == out_values, "The depth values from the sidebar are not converted correctly"

    test([1.0, 200.0, 3.0], [1.0, 2.0, 3.0])
    test([100.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    test([1.0, 2.0, 300.0], [1.0, 2.0, 3.0])
    test([1.0, 200.0, 300.0, 4.0, 5.0, 6.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0])
    test([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])
