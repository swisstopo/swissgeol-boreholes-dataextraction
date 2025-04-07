"""Classifier module."""

import re
from typing import Protocol

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
from stratigraphy.util.util import read_params
from nltk.stem.snowball import SnowballStemmer


classification_params = read_params("classification_params.yml")
language_resources = read_params("language_resources.yml")


class Classifier(Protocol):
    """Classifier Protocol."""

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """


class DummyClassifier:
    """Dummy classifier class.

    Assigns the class USCSClasses.CL_ML to all descriptions
    """

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        for layer in layer_descriptions:
            layer.prediction_uscs_class = USCSClasses.CL_ML


class BaselineClassifier:
    """Baseline classifier class.

    Uses keyword pattern matching with language-specific adjustments:
    - Stemming to normalize words
    - Synonym recognition to handle equivalent terms
    - Stopword filtering to ignore common non-diagnostic words
    - Sequential pattern matching with flexible positioning
    """

    def __init__(self, match_threshold=0.5, partial_match_threshold=0.75):
        """Initialize with configurable thresholds.

        Args:
            match_threshold: Minimum coverage for exact position matches (default: 0.75)
            partial_match_threshold: Minimum coverage for partial position matches (default: 0.75)
        """

        self.match_threshold = match_threshold
        self.partial_match_threshold = partial_match_threshold

        language_resources = read_params("language_resources.yml")
        self.uscs_patterns = classification_params["uscs_patterns"]

        self.synonyms = {
            lang: {base: words for base, words in synonyms.items()}
            for lang, synonyms in language_resources.get("synonyms", {}).items()
        }

        self.stopwords = {
            lang: words for lang, words in language_resources.get("stopwords", {}).items()
        }

        self.stemmer_languages = {
            "de": "german",
            "fr": "french",
            "en": "english",
            "it": "italian"
        }

        self.stemmers = {}

    def is_synonym_match(self, token1, token2, language) -> bool:
        """Checks if two tokens are equal or if they are synonyms in the given language.

        Args:
        token1 (str): The first token to compare
        token2 (str): The second token to compare
        language (str): The language code for which synonym dictionary to use

        Returns:
        bool: True if tokens are considered synonyms, False otherwise
        """

        if token1 == token2:
            return True

        for _, synonyms in self.synonyms.get(language, {}).items():
            if token1 in synonyms and token2 in synonyms:
                return True

        return False

    def filter_stopwords(self, tokens, language) -> list[str]:
        """Remove stopwords from a list of tokens.

        Args:
        tokens (list[str]): List of word tokens to filter
        language (str): Language code for which stopword set to use

        Returns:
        list[str]: A new list containing only the tokens without stopwords
        """

        lang_stopwords = self.stopwords.get(language, [])
        return [token for token in tokens if token not in lang_stopwords]

    def get_stemmer(self, language: str) -> dict[str, SnowballStemmer]:
            """Get or create a stemmer for the specified language with German as a fallback option. 
            As its base it utilizes the nltk package and snowball stemmer.

            Args:
                language (str): The language code for which to get the stemmer

            Returns:
                SnowballStemmer: The stemmer for the specified language or German if language is not supported
            """

            if language not in self.stemmers:
                stemmer_lang = self.stemmer_languages.get(language, "german")
                self.stemmers[language] = SnowballStemmer(stemmer_lang)

            return self.stemmers[language]

    def find_ordered_sequence(self, pattern_tokens, description_tokens, language, match_threshold, partial_match_threshold) -> tuple:
        """Find the best match for pattern tokens within description tokens

        Uses two matching strategies:
        1. Exact sequence matching - pattern is treated as a continuous sequence
        2. Flexible matching - pattern is treated as an order sequence with gaps

        Args:
            pattern_tokens: Stemmed tokens from the pattern
            description_tokens: Stemmed tokens from the description
            language: Language code for synonym matching

        Returns:
            tuple: (coverage, start_pos, matched_words)
                - coverage: Ratio of matched tokens (0.0-1.0)
                - start_pos: Position where match begins (-1 if no match)
                - matched_words: List of tokens that matched the pattern
        """
        if not pattern_tokens:
            return 0, -1, []

        description_len = len(description_tokens)
        pattern_len = len(pattern_tokens)

        # Look for exact sequence matches at each possible starting position in description
        if description_len >= pattern_len:
            for start_pos in range(description_len - pattern_len + 1):
                matches = 0
                matched_words = []

                # Check if pattern matches at this position
                for i, p_token in enumerate(pattern_tokens):
                    d_token = description_tokens[start_pos + i]
                    if self.is_synonym_match(p_token, d_token, language):
                        matches += 1
                        matched_words.append(d_token)

                coverage = matches / pattern_len
                if coverage >= match_threshold:
                    return coverage, start_pos, matched_words

        # Look for partial sequenc matches with flexible position matching
        matches = 0
        matched_words = []
        last_match_pos = -1

        for p_token in pattern_tokens:
            # Look for this pattern token anywhere after the last match
            for d_idx in range(last_match_pos + 1, description_len):
                if self.is_synonym_match(p_token, description_tokens[d_idx], language):
                    matches += 1
                    matched_words.append(description_tokens[d_idx])
                    last_match_pos = d_idx
                    break

            coverage = matches / pattern_len
            if coverage >= partial_match_threshold:
                return coverage, 0, matched_words

        return 0, -1, []


    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method analyzes each layer's material description and assigns a USCS class
        by pattern matching. USCS patterns and material descriptions are tokenized, stemmed
        and stopwords are removed from both lists. The best pattern match is selected based on:
        - Coverage (percentage of pattern words matched) - highest priority
        - Complexity (number of words in pattern) - second priority
        - Position (earlier matches preferred) - third priority

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """

        for layer in layer_descriptions:
            patterns = self.uscs_patterns[layer.language]
            description = layer.material_description.lower()
            language = layer.language
            stemmer = self.get_stemmer(language)

            # Tokenize the description into separate words and stem them
            description_tokens = re.findall(r'\b\w+\b', description.lower())
            description_tokens = self.filter_stopwords(description_tokens, language)
            stemmed_description_tokens = [stemmer.stem(token) for token in description_tokens]

            matches = []

            for class_key, class_keywords in patterns.items():
                uscs_class = map_most_similar_uscs(class_key)
                # Create a regex pattern without the class name in parenthesis
                clean_pattern = re.sub(r"\(.*?\)", "", class_keywords).strip().lower()

                # Tokenize the pattern into separate words and stem them
                pattern_tokens = re.findall(r'\b\w+\b', clean_pattern)
                pattern_tokens = self.filter_stopwords(pattern_tokens, language)
                stemmed_pattern_tokens = [stemmer.stem(token) for token in pattern_tokens]

                coverage, start_pos, matched_words = self.find_ordered_sequence(
                    stemmed_pattern_tokens, stemmed_description_tokens, language,
                    self.match_threshold, self.partial_match_threshold
                )

                matches.append({
                    'class': uscs_class,
                    'coverage': coverage,
                    'complexity': len(pattern_tokens),
                    'matched_words': matched_words,
                    'start_pos': start_pos,
                })

            # Sort matches by coverage and complexity in descending order, then by start_pos in ascending order
            sorted_matches = sorted(matches, key=lambda x: (-x['coverage'], -x['complexity'], x['start_pos']))

            if sorted_matches:
                layer.prediction_uscs_class = sorted_matches[0]['class']
            else:
                layer.prediction_uscs_class = USCSClasses.kA
