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

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        uscs_patterns = classification_params["uscs_patterns"]

        for layer in layer_descriptions:
            patterns = uscs_patterns[layer.language]
            description = layer.material_description.lower()
            detected_class = None

            # Iterate over USCS classes and match with regex
            for class_key, class_keywords in patterns.items():
                # agressive search, will split by keyword and stop at first keyword match
                for keyword in re.sub(r"[()]", "", class_keywords).split():  # remove () around class label
                    if re.search(rf"\b{re.escape(keyword.lower())}\b", description):
                        detected_class = map_most_similar_uscs(class_key)
                        break  # Stop at the first match
                if detected_class:
                    break

            # Assign the detected class or default to "keine Angabe" (kA)
            layer.prediction_uscs_class = detected_class if detected_class else USCSClasses.kA
