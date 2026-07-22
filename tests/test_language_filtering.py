"""Unit tests for language-filtering utilities."""

import pytest

from extraction.features.metadata.borehole_name_extraction import _find_candidate_name, clean_borehole_name
from swissgeol_doc_processing.utils.language_filtering import (
    normalize_spaces,
    remove_any_keyword,
    remove_in_parenthesis,
    remove_scale,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("text", "text"),
        ("text 1:100", "text"),
        ("text 1 : 100", "text"),
        ("text M1:100", "text"),
        ("text M.1:100", "text"),
        ("text M 1:100", "text"),
    ],
    ids=[
        "none",
        "scale-simple",
        "scale-with-spaces",
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
        ("text (parenthesis)", "text"),
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
    text = clean_borehole_name(text, excluded_keywords)
    assert text == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SONDAGE CAROTTÉ S1", "S1"),
        ("Kernbohrung Kb 02/4", "Kb 02/4"),
        ("Bohrung KB1-18/P", "KB1-18/P"),
        ("G6/P", "G6/P"),
        ("Baggerschlitz BS 16-1/P", "BS 16-1/P"),
        ("N°8366", "N°8366"),
    ],
)
def test_findcandidatename(text: str, expected: str | None) -> None:
    """Test borehole name extraction behavior.

    Args:
        text (str): Input string containing the borehole name.
        expected (str | None): The candidate borehole name that should be extracted from the input.
    """
    words = text.split(" ")
    if match := _find_candidate_name(words):
        start, end = match
        name = " ".join(words[start:end])
    else:
        name = None
    assert name == expected
