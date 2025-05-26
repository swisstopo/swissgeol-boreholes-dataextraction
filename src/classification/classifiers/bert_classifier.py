"""Bert-based classifier module."""

from pathlib import Path

import mlflow
import numpy as np
from classification.classifiers.classifier import Classifier
from classification.models.model import BertModel
from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformations
from transformers import Trainer, TrainingArguments


class BertClassifier(Classifier):
    """Classifier class that uses the BERT model."""

    def __init__(self, model_path: Path | None, classification_system: type[ClassificationSystem]):
        """Initialize a BertClassifier instance.

        Args:
            model_path (Path | None): Path to the model checkpoint.
            classification_system (type[ClassificationSystem]): the classification system used to classify
                the descriptions.
        """
        self.init_config(classification_system)
        if model_path is None:
            # load pretrained from transformers lib (bad)
            model_path = self.config["model_path"]
        self.model_path = model_path
        self.bert_model = BertModel(model_path, classification_system)

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "bert"

    def log_params(self):
        """Log the name of the model used."""
        mlflow.log_param("model_name", "/".join(self.model_path.parts[-2:]))

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object
        """
        # We create an instance of Trainer only for prediction as it is much faster than using custom methods.
        eval_dataset = self.bert_model.get_tokenized_dataset(layer_descriptions)
        trainer = Trainer(
            model=self.bert_model.model,
            processing_class=self.bert_model.tokenizer,
            args=TrainingArguments(per_device_eval_batch_size=self.config["inference_batch_size"]),
        )
        output = trainer.predict(eval_dataset)
        predicted_indices = list(np.argmax(output.predictions, axis=1))

        # Convert indices to Enum classes and assign them
        for layer, idx in zip(layer_descriptions, predicted_indices, strict=True):
            layer.prediction_class = self.bert_model.id2classEnum[idx]
