"""Unit tests for language-filtering utilities."""

import pytest

from extraction.features.metadata.borehole_name_extraction import _clean_borehole_name
from swissgeol_doc_processing.utils.language_filtering import (
    match_any_keyword,
    normalize_spaces,
    remove_any_keyword,
    remove_in_parenthesis,
    remove_scale,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("text", "text"),
        ("text 1:100", "text "),
        ("text M1:100", "text "),
        ("text M.1:100", "text "),
        ("text M 1:100", "text "),
    ],
    ids=[
        "none",
        "scale-simple",
        "scale-masstab",
        "scale-masstab-punct",
        "scale-space",
    ],
)
def test_remove_scale(text: str, expected: str) -> None:
    """Verify that `remove_scale` removes scale notations.

    Args:
        text (str): Input text possibly containing a scale pattern.
        expected (str): The expected string after removing the scale pattern.
    """
    assert expected == remove_scale(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("text", "text"),
        ("text (parenthesis)", "text "),
    ],
    ids=[
        "none",
        "parenthesis",
    ],
)
def test_remove_in_parenthesis(text: str, expected: str) -> None:
    """Verify that `remove_in_parenthesis` removes content inside parentheses.

    Args:
        text (str): Input text possibly containing parenthetical content.
        expected (str): The expected string after removal.
    """
    assert expected == remove_in_parenthesis(text)


@pytest.mark.parametrize(
    "text, keywords, start, end, ignore_case, expected",
    [
        ("test schachtprofil 12", ["schachtprofil"], False, False, True, "schachtprofil"),
        ("test Schachtprofil 12", ["schachtprofil"], False, False, True, "Schachtprofil"),
        ("test Schachtprofil 12", ["schachtprofil"], False, False, False, None),
        ("test schachtprofil 12", ["schacht"], True, False, True, "schachtprofil"),
        ("test schachtprofil 12", ["schacht"], False, True, True, None),
        ("test schachtprofil 12", ["profil"], False, True, True, "schachtprofil"),
        ("test schachtprofil 12", ["profil"], True, False, True, None),
        ("test forage schachtprofil 12", ["forage", "profil"], False, True, True, "forage"),
        ("test forage KB12", ["KB ??\d+"], True, True, False, "KB12"),
        ("KB Guatelli KBaBcd12", ["KB ??\d+"], True, True, False, None),
    ],
    ids=[
        "full-word",
        "ignore-case",
        "case-sensitive",
        "anchored-start",
        "neg-anchored-start",
        "anchored-end",
        "neg-anchored-end",
        "first-match",
        "regex-kb-match",
        "regex-no-kb-match-1",
    ],
)
def test_match_any_keyword(
    text: str, keywords: list[str], start: bool, end: bool, ignore_case: bool, expected: str
) -> None:
    """Test keyword search from a predefined list in a text.

    Args:
        text (str): Text to search within.
        keywords (list[str]): Keywords to look for (treated as literals).
        start (bool): If True, the matched word must start with the keyword.
        end (bool): If True, the matched word must end with the keyword.
        ignore_case (bool): If True, keyword matching is case insensitive.
        expected (str): The substring expected to be matched in `text`.
    """
    match = match_any_keyword(text, keywords, start, end, ignore_case)

    if expected:
        assert text[match.start() : match.end()] == expected
    else:
        assert match is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("forage  nr  3", "forage nr 3"),
        ("  sondierbohrung 2  ", "sondierbohrung 2"),
    ],
    ids=["double-space", "leading-trailing"],
)
def test_normalize_spaces(text: str, expected: str) -> None:
    """Test text space nromalization.

    Args:
        text (str): Text to normalize.
        expected (str): Normalized output.
    """
    text = normalize_spaces(text)
    assert text == expected


@pytest.mark.parametrize(
    "text, keywords, expected",
    [
        ("nr1", ["nr"], "1"),
        ("nr 1", ["nr"], " 1"),
        ("n r 1", ["n r"], " 1"),
        ("n.r 1", ["n.r"], " 1"),
        ("Nr 1", ["nr"], " 1"),
        ("sondage nº1", ["nº"], "sondage 1"),
        ("sondage n°1", ["n°"], "sondage 1"),
        ("sondage Nr nº 1", ["nr", "nº"], "sondage   1"),
    ],
    ids=[
        "nr-without-space",
        "nr-with-space",
        "nr-with-space2",
        "nr-point",
        "ignore-case",
        "n-masc-ordinal",
        "n-degree",
        "multiple-keywords",
    ],
)
def test_remove_any_keyword(text: str, keywords: list[str], expected: str) -> None:
    """Test keyword removal in text.

    Note: Expected strings here intentionally preserve existing spacing so we can
    verify that only the keywords are removed and no extra normalization occurs.

    Args:
        text (str): Input text to filter.
        keywords (list[str]): List of keywords to filter out.
        expected (str): Expected filtered text.
    """
    text = remove_any_keyword(text, keywords)
    assert text == expected


@pytest.mark.parametrize(
    "text, excluded_keywords, expected",
    [
        ("schachtprofil 12", [], "schachtprofil 12"),
        ("schachtprofil 12", None, "schachtprofil 12"),
        ("n r nr. schachtprofil nr-12", ["schachtprofil", "nr.", "n r"], "nr-12"),
        ("SP1 1:20", [], "SP1"),
        ("SP1 (comment)", [], "SP1"),
        ("schachtprofil.:_ 12", [], "schachtprofil 12"),
        ("", [], None),
    ],
    ids=[
        "empty-keywords",
        "none-keywords",
        "exclude-keywords",
        "exclude-scale",
        "exclude-parenthesis",
        "exclude-punc",
        "exclude-empty",
    ],
)
def test_clean_borehole_name(text: str, excluded_keywords: list[str], expected: str | None) -> None:
    """Test borehole name cleaning behavior.

    Args:
        text (str): Input string containing the borehole name.
        excluded_keywords (list[str]): Keywords to strip from the name.
        expected (str | None): The cleaned substring that should be matched.
    """
    text = _clean_borehole_name(text, excluded_keywords)
    assert text == expected
