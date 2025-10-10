"""Unit tests for language-filtering utilities."""

import pytest

from utils.language_filtering import match_any_keyword, normalize_spaces, remove_any_keyword


@pytest.mark.parametrize(
    "text, keywords, start, end, expected",
    [
        ("test schachtprofil 12", ["schachtprofil"], False, False, "schachtprofil"),
        ("test schachtprofil 12", ["schacht"], True, False, "schachtprofil"),
        ("test schachtprofil 12", ["profil"], False, True, "schachtprofil"),
        ("test forage schachtprofil 12", ["forage", "profil"], False, True, "forage"),
    ],
    ids=[
        "full-word",
        "anchored-start",
        "anchored-end",
        "first-match",
    ],
)
def test_match_any_keyword(text: str, keywords: list[str], start: bool, end: bool, expected: str) -> None:
    """Test keyword search from a predefined list in a text.

    Args:
        text (str): Text to search within.
        keywords (list[str]): Keywords to look for (treated as literals).
        start (bool): If True, the matched word must start with the keyword.
        end (bool): If True, the matched word must end with the keyword.
        expected (str): The substring expected to be matched in `text`.
    """
    match = match_any_keyword(text, keywords, start, end)
    assert match is not None
    assert text[match.start() : match.end()] == expected


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
        ("sondage nº1", ["nº"], "sondage 1"),
        ("sondage n°1", ["n°"], "sondage 1"),
    ],
    ids=["nr-without-space", "nr-with-space", "n-masc-ordinal", "n-degree"],
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
