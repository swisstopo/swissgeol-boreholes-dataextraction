"""Classifier module."""

from abc import ABC, abstractmethod
from enum import Enum

from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformations
from utils.file_utils import read_params

CONFIG_MAPINGS = read_params("classifier_config_paths.yml")


class Classifier(ABC):
    """Classifier Protocol."""

    def init_config(self, classification_system: type[ClassificationSystem]):
        """Initialize the model config dict based on the classification system.

        Args:
            classification_system (type[ClassificationSystem]): The classification system used.
        """
        config_file = CONFIG_MAPINGS[self.get_name()][classification_system.get_name()]
        self.config = read_params(config_file) if config_file else None

    @abstractmethod
    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        raise NotImplementedError

    @abstractmethod
    def log_params(self):
        """Log the parameters of the classifier, if any."""
        raise NotImplementedError

    @abstractmethod
    def classify(self, layer_descriptions: list[LayerInformations]) -> None:
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        raise NotImplementedError


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
