"""Test the TextLine class matching and compound splitting functionality."""

import pytest

from extraction.features.utils.text.stemmer import _split_compounds, find_matching_expressions, _match_patterns


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
    "patterns, targets, expected",
    [
        (["sand", "argile", "rock"], ["argile"], ["argile"]),
        (["rock"], ["sand", "argile", "rock"], ["rock"]),
        (["sand", "argile", "rock"], ["argile", "rock"], ["argile", "rock"]),
        (["sand", "argile", "rock"], ["argile", "argile"], ["argile", "argile"]),
        (["sand", "argile"], ["rock"], []),
        (["rock"], ["sand", "argile"], []),
    ],
    ids=["many-to-one", "one-to-many", "many-to-many", "redunduncy", "many-to-none", "none-to-many"],
)
def test_match_patterns(patterns: list[str], targets: list[str], expected: list[str]) -> None:
    """Test that `_match_patterns` correctly identifies matching elements between patterns and targets.

    Args:
        patterns (list[str]): The list of pattern strings to search for.
        targets (list[str]): The list of target strings to check against.
        expected (list[str]): The expected list of matched patterns.
    """
    result = _match_patterns(patterns, targets)
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
    assert result == expected
