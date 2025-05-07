"""Dummy classifier module."""

from classification.utils.data_loader import LayerInformations


class DummyClassifier:
    """Dummy classifier class.

    Assigns the most common class to all descriptions
    """

    def classify(self, layer_descriptions: list[LayerInformations]) -> None:
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): List of layer information objects to classify.
        """
        for layer in layer_descriptions:
            layer.prediction_class = layer.class_system.get_dummy_classifier_class_value()
