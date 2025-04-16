"""Baseline classifier module."""

import re

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
from nltk.stem.snowball import SnowballStemmer
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")


class BaselineClassifier:
    """Baseline classifier class.

    The BaselineClassifier works by matching stemmed USCS patterns against layer descriptions using
    a flexible ordered sequence matching algorithm.
    """

    def __init__(self, match_threshold=0.75):
        """Initialize with configurable threshold.

        Args:
            match_threshold (float): Minimum coverage for matches (default: 0.75)
        """
        self.match_threshold = match_threshold

        self.uscs_patterns = classification_params["uscs_patterns"]

        self.stemmer_languages = {"de": "german", "fr": "french", "en": "english", "it": "italian"}

        self.stemmers = {}

    def get_stemmer(self, language: str) -> SnowballStemmer:
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

    def find_ordered_sequence(self, pattern_tokens, description_tokens, match_threshold) -> tuple | None:
        """Find the best match for pattern tokens within description tokens.

        This method searches for pattern tokens within description tokens in sequential order,
        allowing for discontinuous matches (matching allows for gaps between pattern tokens).

        Args:
            pattern_tokens (list): List of tokens to search for.
            description_tokens (list): List of tokens to search within.
            match_threshold (float): Minimum coverage ratio required for a match to be valid.

        Returns:
            tuple| None: If a match is found, returns a tuple with:
                - coverage (float): Ratio of matched pattern tokens to total pattern tokens
                - matched_positions (tuple): Positions of matches in description_tokens
                - matched_words (list): The actual matched words
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
        if coverage >= match_threshold:
            return coverage, tuple(matched_positions), matched_words

        return None

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the material descriptions of layer information objects into USCS soil classes.

        The method modifies the input object, layer_descriptions by setting their prediction_uscs_class attribute.
        The approach is as follows:

        1. Tokenize and stem both the material description and the USCS pattern keywords
        2. Find matches between description and patterns using partial matching
        3. Scores matches based on three criteria (in priority order):
           - Coverage: Percentage of pattern words matched in the description
           - Complexity: Length/specificity of the pattern (longer patterns preferred)
           - Position: Earlier matches in the text are preferred
        4. Assigns the best matching USCS class to the layer object

        For layers with no matches, assigns the default class 'kA' (no classification).

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
                        self.match_threshold,
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

            # Sort matches by coverage and complexity in descending order, then by match_positions in ascending order
            sorted_matches = sorted(matches, key=lambda x: (-x["coverage"], -x["complexity"], x["match_positions"]))

            if sorted_matches:
                layer.prediction_uscs_class = sorted_matches[0]["class"]
            else:
                layer.prediction_uscs_class = USCSClasses.kA
