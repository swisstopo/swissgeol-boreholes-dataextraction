"""Classifier module."""

import re
from typing import Protocol

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
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

    Performs a key-word matching to assign the classes.
    """

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object. The layer's soil description is
        matched with the description of the USCS class. For example the class SW-SC has for description "sable bien
        gradué avec argile (SW-SC)" in the classification_params.yml file, so it will try to identify the pattern
        "sable bien gradué avec argile" in the layer's descriptions.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        uscs_patterns = classification_params["uscs_patterns"]

        # Create a stemming function that takes predefined suffix elements for a specific language and stemms all the words
        # We ensure that the word is long enough by checking if the length is greater than the suffix length + 2
        suffix_replacements = {
        "de": [("er", ""), ("en", ""), ("es", ""), ("e", ""), ("s", "")],
        "fr": [("s", ""), ("x", ""), ("aux", "al"), ("eux", "eu"), ("ers", "er")]
        }

        def simple_stem(word: str, language: str) -> str:
            """Simple stemming function to handle common declensions"""
            word = word.lower()
            if language in suffix_replacements:
                for suffix, replacement in suffix_replacements[language]:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:  
                        return word[:-len(suffix)] + replacement
            return word

        for layer in layer_descriptions:
            patterns = uscs_patterns[layer.language]
            description = layer.material_description.lower()
            language = layer.language

            # Tokenize the description into separate words and stem them
            description_tokens = re.findall(r'\b\w+\b', description.lower())
            stemmed_description_tokens = [simple_stem(token, language) for token in description_tokens]

            matches = []

            for class_key, class_keywords in patterns.items():
                uscs_class = map_most_similar_uscs(class_key)
                # Create a regex pattern without the class name in parenthesis
                clean_pattern = re.sub(r"\(.*?\)", "", class_keywords).strip().lower()

                # Tokenize the pattern into separate words and stem them
                pattern_tokens = re.findall(r'\b\w+\b', clean_pattern)
                stemmed_pattern_tokens = [simple_stem(token, language) for token in pattern_tokens]

                matched_tokens = 0
                matched_words = []

                for p_token in stemmed_pattern_tokens:
                    if p_token in stemmed_description_tokens:
                        matched_tokens += 1
                        idx = stemmed_description_tokens.index(p_token)
                        matched_words.append(description_tokens[idx])

                # Calculate coverage score
                if pattern_tokens:
                    coverage = matched_tokens / len(pattern_tokens)

                    # Only consider matches with significant coverage At least 50% of pattern
                    if coverage >= 0.5:
                        matches.append({
                            'class': uscs_class,
                            'coverage': coverage,
                            'complexity': len(pattern_tokens),
                            'matched_words': matched_words
                        })

            # Sort matches by coverage (primary) and complexity (secondary)
            sorted_matches = sorted(matches, key=lambda x: (x['coverage'], x['complexity']), reverse=True)

            # Assign the best match or default to "keine Angabe" (kA)
            if sorted_matches:
                layer.prediction_uscs_class = sorted_matches[0]['class']
            else:
                layer.prediction_uscs_class = USCSClasses.kA
