"""Test suite for the LayerIdentifierSidebar module."""

import pytest
from extraction.extraction.stratigraphy.base.sidebar_entry import LayerIdentifierEntry
from extraction.extraction.stratigraphy.sidebar.classes.layer_identifier_sidebar import (
    LayerIdentifierSidebar,
)


@pytest.mark.parametrize(
    "entries,expected",
    [
        (["1)", "2)", "3)", "4)"], True),  # Pure numeric
        (["a)", "b)", "c)"], True),  # Pure alphabet
        (["1)", "2)", "3a)", "3b)", "4)", "5)"], True),  # Valid mixed
        (["1)", "2)", "2a)", "2b)", "3)", "4a)", "4b)", "5)"], True),  # Complex valid
        (["1)", "2)", "1)", "3)", "4)"], True),  # Valid but with 1 intervals invalid, but still bigger than ratio
        (["1)", "e)", "3)", "4)", "5)", "6)"], True),  # Valid, simulate wrong extraction, but still bigger than ratio
        (["2a2)", "2a4)", "3a)"], True),  # valid, with key of depth 3
        (["1)", "2)", "12)"], True),  # valid, why the conversion to int is needed
        (["1)", "3)", "2)"], False),  # Out-of-order
        (["1)", "1)", "1)"], False),  # No progression
        (["m)", "m)", "m)"], False),  # Simulate the case where the letter m) standing for meter is detected.
        (["1)", "2)", "2)", "2)", "3)", "3)", "4)"], False),  # Too many invalid transitions
        (["2a2)", "2a1)", "3a)"], False),  # invalid, with key of depth 3
    ],
)
def test_has_regular_progression(entries, expected):
    """Test the has_regular_progression method of the LayerIdentifierSidebar.

    It detects if a layer identifier sidebar, like 1), 2), 3) is valid or not. This test is usefull because some
    entries in the document might look like sidebar but actually are not.
    """
    sidebar = LayerIdentifierSidebar([LayerIdentifierEntry(rect=None, value=value) for value in entries])
    assert sidebar.has_regular_progression() == expected
