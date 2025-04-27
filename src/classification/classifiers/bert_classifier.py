"""Bert-based classifier module."""

from pathlib import Path

import numpy as np
from classification.models.model import BertModel
from classification.utils.data_loader import LayerInformations
from transformers import Trainer, TrainingArguments
from utils.file_utils import read_params

model_config = read_params("bert_config.yml")


class BertClassifier:
    """Classifier class that uses the BERT model."""

    def __init__(self, model_path: Path | None):
        if model_path is None:
            # load pretrained from transformers lib (bad)
            model_path = model_config["model_path"]
        self.model_path = model_path
        self.bert_model = BertModel(model_path)

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object
        """
        eval_dataset = self.bert_model.get_tokenized_dataset(layer_descriptions)
        trainer = Trainer(
            model=self.bert_model.model,
            processing_class=self.bert_model.tokenizer,
            args=TrainingArguments(per_device_eval_batch_size=model_config["inference_batch_size"]),
        )
        output = trainer.predict(eval_dataset)
        predicted_indices = list(np.argmax(output.predictions, axis=1))

        # Convert indices to USCSClasses and assign them
        for layer, idx in zip(layer_descriptions, predicted_indices, strict=True):
            layer.prediction_uscs_class = self.bert_model.id2classEnum[idx]
