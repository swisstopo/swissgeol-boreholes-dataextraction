"""Test the TextLine class matching and compound splitting functionality."""

import pytest

from extraction.features.utils.text.stemmer import _split_compounds, find_matching_expressions


@pytest.mark.parametrize(
    "tokens, split_threshold, expected",
    [
        (["Sandstein"], 0.4, ["Sand", "Stein"]),
        (["Kalkstein"], 0.4, ["Kalk", "Stein"]),
        # Edge cases
        ([], 0.4, []),
        ([" "], 0.4, [" "]),
        (["Hallo"], 0.4, ["Hallo"]),
        (["Sandstein"], 0.9, ["Sandstein"]),  # high threshold
        (["Sandstein", "Kalkstein"], 0.4, ["Sand", "Stein", "Kalk", "Stein"]),
    ],
)
def test_split_compounds(tokens, split_threshold, expected):
    """Test the _split_compounds method with various inputs."""
    result = _split_compounds(tokens, split_threshold)
    assert result == expected


@pytest.mark.parametrize(
    "patterns, split_threshold, targets, language, search_excluding, expected",
    [
        (["sand"], 0.4, ["Sandstein"], "de", False, True),
        (["argile"], 0.4, ["argileuses"], "fr", False, True),
        (["rock"], 0.4, ["Rocks"], "en", False, True),
        # Edge cases
        ([], 0.4, ["anything"], "de", False, False),
        (["pattern"], 0.4, [], "de", False, False),
        ([""], 0.4, [""], "de", False, True),
        (["STEIN"], 0.4, ["Sandstein"], "de", False, True),
        (["sand", "kalk"], 0.4, ["Sandstein", "Kalkstein"], "de", False, True),
        (["sand"], 0.4, ["Sandstein"], "de", True, False),  # Excluding match
        (["sand"], 0.9, ["Sandstein"], "de", False, False),  # high threshold
    ],
)
def test_find_matching_expressions(patterns, split_threshold, targets, language, search_excluding, expected):
    """Test the _find_matching_expressions method with various inputs."""
    result = find_matching_expressions(
        patterns, split_threshold, targets, language, analytics=None, search_excluding=search_excluding
    )
    print(expected)
    assert result == expected
