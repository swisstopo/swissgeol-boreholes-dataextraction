"""Test suite for the  Protocol sidebar."""

import pymupdf

from extraction.features.stratigraphy.base.sidebar_entry import DepthColumnEntry
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import (
    ProtocolSidebar,
)


def test_trim_trailing_duplicate_depths():
    """Test the trim_trailing_duplicate_depths method of the ProtocolSidebar class."""
    sidebar = ProtocolSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=5, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=155, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=155, page_number=0),
        ]
    )
    sidebar = sidebar.trim_trailing_duplicate_depths()
    assert ProtocolSidebar(
        [
            DepthColumnEntry(rect=pymupdf.Rect(), value=5, page_number=0),
            DepthColumnEntry(rect=pymupdf.Rect(), value=155, page_number=0),
        ]
    )
