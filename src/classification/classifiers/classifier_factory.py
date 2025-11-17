"""Factory that produce the correct type of classifier."""

import logging
from pathlib import Path

from classification.classifiers.aws_bedrock_classifier import AWSBedrockClassifier
from classification.classifiers.baseline_classifier import BaselineClassifier
from classification.classifiers.bert_classifier import BertClassifier
from classification.classifiers.classifier import ClassifierTypes
from classification.classifiers.dummy_classifier import DummyClassifier
from classification.utils.classification_classes import ClassificationSystem

logger = logging.getLogger(__name__)


class ClassifierFactory:
    """Classifier Factory to procduce the correct classifier based on the type."""

    @staticmethod
    def create_classifier(
        classifier_type: ClassifierTypes,
        classification_system: type[ClassificationSystem],
        model_path: Path,
        out_directory_bedrock: Path,
    ):
        """Factory method to create a classifier instance.

        Args:
            classifier_type (ClassifierTypes): Type of classifier to create.
            classification_system (type[ClassificationSystem]): The classification system to be used.
            model_path (Path): Path to the model (used for BERT).
            out_directory_bedrock (Path): Output directory for Bedrock classifier.

        Returns:
            A classifier instance.
        """
        if model_path is not None and classifier_type != ClassifierTypes.BERT:
            logger.warning("Model path is only used with classifier 'bert'.")

        if classifier_type == ClassifierTypes.DUMMY:
            return DummyClassifier()
        elif classifier_type == ClassifierTypes.BASELINE:
            return BaselineClassifier(classification_system)
        elif classifier_type == ClassifierTypes.BERT:
            return BertClassifier(model_path, classification_system)
        elif classifier_type == ClassifierTypes.BEDROCK:
            return AWSBedrockClassifier(out_directory_bedrock, classification_system)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
