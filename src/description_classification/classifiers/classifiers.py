"""Classifier module."""

from dataclasses import dataclass
from typing import Protocol

from description_classification.utils.data_loader import LayerDescription
from description_classification.utils.uscs_classes import USCSClasses
from stratigraphy.util.util import read_params

classification_params = read_params("classification_params.yml")


@dataclass
class LayerUSCSPrediction:
    """Da."""

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    uscs_class: USCSClasses


class Classifier(Protocol):
    """Classifier Protocol."""

    def classify(self, descriptions: list[LayerDescription]) -> list[LayerUSCSPrediction]:
        """Classify.

        Args:
            descriptions (list[LayerDescription]): _description_

        Returns:
            list[LayerUSCSPrediction]: _description_
        """


class DummyClassifier:
    """Dummy classifier class."""

    def classify(self, descriptions: list[LayerDescription]) -> list[LayerUSCSPrediction]:
        """Classify.

        Args:
            descriptions (list[LayerDescription]): _description_

        Returns:
            list[LayerUSCSPrediction]: _description_
        """
        predictions = []
        for descr in descriptions:
            predictions.append(
                LayerUSCSPrediction(
                    descr.filename, descr.borehole_index, descr.layer_index, descr.language, USCSClasses.CL_ML
                )
            )
        return predictions


class BaselineClassifier:
    """Baseline classifier class."""

    def classify(self, descriptions: list[LayerDescription]) -> list[LayerUSCSPrediction]:
        """Classify.

        Args:
            descriptions (list[LayerDescription]): _description_

        Returns:
            list[LayerUSCSPrediction]: _description_
        """
        # uscs_patterns = classification_params["uscs_patterns"]

        raise NotImplementedError
