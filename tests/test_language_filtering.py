"""Unit tests for language-filtering utilities."""

import pytest

from extraction.features.metadata.borehole_name_extraction import _find_candidate_names, clean_borehole_name
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.language_filtering import (
    normalize_spaces,
    remove_any_keyword,
)


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
        ("schachtprofil.:_ 12", [], "schachtprofil 12"),
        ("", [], None),
    ],
    ids=[
        "empty-keywords",
        "none-keywords",
        "exclude-keywords",
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


name_detection_params = read_params("name_detection_params.yml")
excluded_keywords: list[str] = name_detection_params.get("excluded_keywords")


@pytest.mark.parametrize(
    "text, expected, allow_simple",
    [
        ("SONDAGE CAROTTÉ S1", ["S1"], False),
        ("Kernbohrung Kb 02/4", ["Kb 02/4"], False),
        ("Bohrung KB1-18/P", ["KB1-18/P"], False),
        ("G6/P", ["G6/P"], False),
        ("Baggerschlitz BS 16-1/P", ["BS 16-1/P"], False),
        ("N°8366", ["N°8366"], False),
        ("1 /82", [], False),
        ("1 /82", ["1 /82"], True),
        ("Nr.8", ["Nr.8"], False),
        ("Nr. 102", [], False),
        ("Nr. 102", ["102"], True),
        ("Datum:9.2.81 Sondierung No. KR.1", ["Datum:9.2.81", "KR.1"], False),
        ("571112/256198", [], False),
    ],
)
def test_findcandidatename(text: str, expected: str | None, allow_simple: bool) -> None:
    """Test borehole name extraction behavior."""
    words = text.split(" ")
    names = []
    for start, end in _find_candidate_names(words, excluded_keywords, allow_simple):
        names.append(" ".join(words[start:end]))
    assert names == expected
