"""Dummy classifier module."""

from classification.classifiers.classifier import Classifier
from classification.utils.datasets.classification import LayerInformation


class DummyClassifier(Classifier):
    """Dummy classifier class.

    Assigns the most common class to all descriptions
    """

    def classify(self, layer_descriptions: list[LayerInformation]) -> list[LayerInformation]:
        """Classifies the description of the LayerInformation objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformation]): List of layer information objects to classify.

        Return:
            layer_descriptions (list[LayerInformation]): List of updated objects.
        """
        for layer in layer_descriptions:
            layer.prediction_class = layer.class_system.get_default_class_value()

        return layer_descriptions

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "dummy"

    def log_params(self):
        """No parameters to log."""
        return
