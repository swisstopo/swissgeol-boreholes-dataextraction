"""Classifier module."""

from enum import Enum
from typing import Protocol

from classification.utils.data_loader import LayerInformations


class Classifier(Protocol):
    """Classifier Protocol."""

    def classify(self, layer_descriptions: list[LayerInformations]) -> None:
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        ...

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        ...

    def log_params(self):
        """Log the parameters of the classifier, if any."""
        ...


class ClassifierTypes(Enum):
    """Enum class for all the availlable classifier types."""

    BASELINE = "baseline"
    BERT = "bert"
    BEDROCK = "bedrock"
    DUMMY = "dummy"

    @classmethod
    def infer_type(cls, classifier_str) -> "ClassifierTypes":
        """Infer the classifier type from the string.

        Args:
            classifier_str (str): The classifier type as a string.

        Returns:
            ClassifierTypes: The corresponding ClassifierTypes enum value.
        """
        for classifier in cls:
            if classifier.value == classifier_str:
                return classifier
        raise ValueError(
            f"Invalid classifier type : {classifier_str}, chose from {[classifier.value for classifier in cls]}"
        )
