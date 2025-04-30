"""Dummy classifier module."""

from classification.utils.data_loader import LayerInformations
from classification.utils.lithology_classes import LithologyClasses
from classification.utils.uscs_classes import USCSClasses


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
            if layer.data_type == "uscs":
                layer.prediction_class = USCSClasses.CL_ML
            else:
                layer.prediction_class = LithologyClasses.Marlstone
