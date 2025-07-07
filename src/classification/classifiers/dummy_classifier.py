"""Dummy classifier module."""

from classification.classifiers.classifier import Classifier
from classification.utils.data_loader import LayerInformation


class DummyClassifier(Classifier):
    """Dummy classifier class.

    Assigns the most common class to all descriptions
    """

    def classify(self, layer_descriptions: list[LayerInformation]) -> None:
        """Classifies the description of the LayerInformation objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformation]): List of layer information objects to classify.
        """
        for layer in layer_descriptions:
            layer.prediction_class = layer.class_system.get_dummy_classifier_class_value()

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "dummy"

    def log_params(self):
        """No parameters to log."""
        return
