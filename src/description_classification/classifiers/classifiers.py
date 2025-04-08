"""Classifier module."""

import re
from typing import Protocol

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
from nltk.stem.snowball import SnowballStemmer
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")


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
            match_threshold: Minimum coverage for exact position matches (default: 0.5)
            partial_match_threshold: Minimum coverage for partial position matches (default: 0.75)
        """
        self.match_threshold = match_threshold
        self.partial_match_threshold = partial_match_threshold

        self.uscs_patterns = classification_params["uscs_patterns"]

        self.stemmer_languages = {"de": "german", "fr": "french", "en": "english", "it": "italian"}

        self.stemmers = {}

    def get_stemmer(self, language: str) -> dict[str, SnowballStemmer]:
        """Get or create a stemmer for the specified language with German as a fallback option.

        Args:
            language (str): The language code for which to get the stemmer

        Returns:
            SnowballStemmer: The stemmer for the specified language
        """
        if language not in self.stemmers:
            stemmer_lang = self.stemmer_languages.get(language, "german")
            self.stemmers[language] = SnowballStemmer(stemmer_lang)

        return self.stemmers[language]

    def find_ordered_sequence(self, pattern_tokens, description_tokens, partial_match_threshold) -> tuple | None:
        """Find the best match for pattern tokens within description tokens.

        Uses two matching strategies:
        1. Exact sequence matching - pattern is treated as a continuous sequence
        2. Flexible matching - pattern is treated as an order sequence with gaps

        Args:
            pattern_tokens: Stemmed tokens from the pattern
            description_tokens: Stemmed tokens from the description
            partial_match_threshold: Minimum coverage for partial position matches

        Returns:
            tuple | None: (coverage, start_pos, matched_words)
                - coverage: Ratio of matched tokens (0.0-1.0)
                - match_positions: Positions of the matched tokens in the description_tokens as a tuple
                - matched_words: List of tokens that matched the pattern
        """
        if not pattern_tokens:
            return None

        description_len = len(description_tokens)
        pattern_len = len(pattern_tokens)

        # Look for partial sequence matches with flexible position matching
        matched_words = []
        last_match_pos = -1
        matched_positions = []

        for p_token in pattern_tokens:
            # Look for this pattern token anywhere after the last match
            for d_idx in range(last_match_pos + 1, description_len):
                if p_token == description_tokens[d_idx]:
                    matched_positions.append(d_idx)
                    matched_words.append(description_tokens[d_idx])
                    last_match_pos = d_idx
                    break

        coverage = len(matched_positions) / pattern_len
        if coverage >= partial_match_threshold:
            return coverage, tuple(matched_positions), matched_words

        return None

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
            description_tokens = re.findall(r"\b\w+\b", description.lower())
            stemmed_description_tokens = [stemmer.stem(token) for token in description_tokens]

            matches = []

            for class_key, class_keyphrases in patterns.items():
                uscs_class = map_most_similar_uscs(class_key)
                for class_keyphrase in class_keyphrases:
                    # Tokenize the pattern into separate words and stem them
                    pattern_tokens = re.findall(r"\b\w+\b", class_keyphrase)
                    stemmed_pattern_tokens = [stemmer.stem(token) for token in pattern_tokens]

                    result = self.find_ordered_sequence(
                        stemmed_pattern_tokens,
                        stemmed_description_tokens,
                        self.partial_match_threshold,
                    )

                    if result:
                        coverage, match_positions, matched_words = result
                        matches.append(
                            {
                                "class": uscs_class,
                                "coverage": coverage,
                                "complexity": len(pattern_tokens),
                                "matched_words": matched_words,
                                "match_positions": match_positions,
                            }
                        )

            # Sort matches by coverage and complexity in descending order, then by start_pos in ascending order
            sorted_matches = sorted(matches, key=lambda x: (-x["coverage"], -x["complexity"], x["match_positions"]))

            if sorted_matches:
                layer.prediction_uscs_class = sorted_matches[0]["class"]
            else:
                layer.prediction_uscs_class = USCSClasses.kA
