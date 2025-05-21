"""Bert-based classifier module."""

from pathlib import Path

import numpy as np
from classification.models.model import BertModel
from classification.utils.data_loader import LayerInformations
from transformers import Trainer, TrainingArguments
from utils.file_utils import read_params


class BertClassifier:
    """Classifier class that uses the BERT model."""

    def __init__(self, model_path: Path | None, classification_system_str: str):
        """Initialize a BertClassifier instance.

        Args:
            model_path (Path | None): Path to the model checkpoint.
            classification_system_str (str): the classification system used to classify the descriptions.
        """
        self.init_model_config(classification_system_str)
        if model_path is None:
            # load pretrained from transformers lib (bad)
            model_path = self.model_config["model_path"]
        self.model_path = model_path
        self.bert_model = BertModel(model_path, classification_system_str)

    def init_model_config(self, classification_system_str: str):
        """Initialize the model config dict based on the classification system.

        Args:
            classification_system_str (str): The classification system used (`uscs` or `lithology`).
        """
        config_file = "bert_config_uscs.yml" if classification_system_str == "uscs" else "bert_config_lithology.yml"
        self.model_config = read_params(config_file)

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
            args=TrainingArguments(per_device_eval_batch_size=self.model_config["inference_batch_size"]),
        )
        output = trainer.predict(eval_dataset)
        predicted_indices = list(np.argmax(output.predictions, axis=1))

        # Convert indices to Enum classes and assign them
        for layer, idx in zip(layer_descriptions, predicted_indices, strict=True):
            layer.prediction_class = self.bert_model.id2classEnum[idx]
