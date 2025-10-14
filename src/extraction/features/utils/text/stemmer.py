"""This module provides stemmer implementation for text processing."""

from compound_split import char_split
from nltk.stem.snowball import SnowballStemmer

from extraction.features.utils.text.matching_params_analytics import MatchingParamsAnalytics, track_match

_LANGUAGE_MAPPING = {
    "de": "german",
    "fr": "french",
    "en": "english",
    "it": "italian",
}
_DEFAULT_LANGUAGE = "german"
_stemmers: dict[str, SnowballStemmer] = {}


def _get_stemmer(language: str) -> SnowballStemmer:
    """Return a cached stemmer instance for the given language.

    This is done in order to create only one stemmer instance of each language during the whole program.

    Args:
        language(str): the language for which the stemmer instance should be returned or created.
    """
    if language not in _stemmers:
        stemmer_lang = _LANGUAGE_MAPPING.get(language, _DEFAULT_LANGUAGE)
        _stemmers[language] = SnowballStemmer(stemmer_lang)
    return _stemmers[language]


def _split_compounds(tokens: list[str], split_threshold: float) -> list[str]:
    """Split compound words using char_split and return processed list.

    This method uses  an ngram-based compound splitter for German language based on
    Tuggener, Don (2016):  https://pypi.org/project/compound-split/

    Args:
        tokens (List[str]): List of tokens to process.
        split_threshold (float): Threshold for splitting compounds

    Returns:
        List[str]: Processed list of tokens with compounds split.
    """
    processed_tokens = []
    for token in tokens:
        comp_split = char_split.split_compound(token)[0]
        if comp_split[0] > split_threshold:
            processed_tokens.extend(comp_split[1:])
        else:
            processed_tokens.append(token)

    return processed_tokens


def _match_patterns(patterns: list[str], targets: list[str]) -> list[str]:
    """Return all patterns that match any of the given targets.

    This function checks whether any of the provided patterns are present
    in the list of targets.

    Args:
        patterns (list[str]): Patterns to search for.
        targets (list[str]): Target strings to check against.

    Returns:
        list[str]: A list of patterns that were found in the targets.
        The list is empty if no matches are found.
    """
    matches = []
    for target in targets:
        if target in patterns:
            matches.append(target)
    return matches


def find_matching_expressions(
    patterns: list,
    split_threshold: float,
    targets: list[str],
    language: str,
    analytics: MatchingParamsAnalytics | None = None,
    search_excluding: bool = False,
) -> bool:
    """Check if any of the patterns match the targets for german use a second check against compound split.

    Args:
        patterns (List): A list of patterns to match against.
        split_threshold (float): Threshold for splitting compounds.
        targets (List[str]): A list of target strings to match against.
        language (str): The language of the patterns, used for stemming.
        analytics ([MatchingParamsAnalytics]): Analytics instance to track matches.
        search_excluding (bool): Whether this is for excluding expressions (for analytics).

    Returns:
        bool: True if any pattern matches, False otherwise.
    """
    stemmer = _get_stemmer(language)

    patterns = {stemmer.stem(p.lower()) for p in patterns}
    targets_stemmed = {stemmer.stem(t.lower()) for t in targets}

    if language == "de" and not search_excluding:
        targets_split = _split_compounds(targets, split_threshold)
        targets_split = {stemmer.stem(t.lower()) for t in targets_split}

        targets_to_check = targets_stemmed | targets_split
    else:
        targets_to_check = targets_stemmed

    # Look for any target that is part of patterns
    target_matches = _match_patterns(patterns, targets_to_check)

    # Update match traking
    [track_match(analytics, m, language, search_excluding) for m in target_matches]

    return len(target_matches) != 0
