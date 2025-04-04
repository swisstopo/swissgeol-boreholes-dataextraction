"""Classifier module."""

import logging
import os
import re
from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
from description_classification.models.model import BertModel
from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
from stratigraphy.util.util import read_params
from transformers import Trainer

logger = logging.getLogger(__name__)

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

classification_params = read_params("classification_params.yml")
model_config = read_params("bert_config.yml")


class Classifier(Protocol):
    """Classifier Protocol."""

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """


class DummyClassifier:
    """Dummy classifier class.

    Assigns the class USCSClasses.CL_ML to all descriptions
    """

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        for layer in layer_descriptions:
            layer.prediction_uscs_class = USCSClasses.CL_ML


class BaselineClassifier:
    """Baseline classifier class.

    Performs a key-word matching to assign the classes.
    """

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object. The layer's soil description is
        matched with the description of the USCS class. For example the class SW-SC has for description "sable bien
        gradué avec argile (SW-SC)" in the classification_params.yml file, so it will try to identify the pattern
        "sable bien gradué avec argile" in the layer's descriptions.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object

        """
        uscs_patterns = classification_params["uscs_patterns"]

        for layer in layer_descriptions:
            patterns = uscs_patterns[layer.language]
            description = layer.material_description.lower()
            detected_class: USCSClasses | None = None

            all_matches: dict[USCSClasses:str] = dict()

            # Iterate over USCS classes and match with regex
            for class_key, class_keywords in patterns.items():
                uscs_class = map_most_similar_uscs(class_key)
                # Create a regex pattern without the class name in parenthesis
                regex_pattern = r"\b" + re.escape(re.sub(r"\(.*?\)", "", class_keywords).strip()) + r"\b"

                # Search for an exact match of the full pattern
                match = re.search(regex_pattern, description)
                if match:
                    matched_text = match.group(0)
                    all_matches[uscs_class] = matched_text

            longest_match: str | None = None
            for matched_class, matched_text in all_matches.items():
                if longest_match is None or len(matched_text.split()) > len(longest_match.split()):
                    longest_match = matched_text
                    detected_class = matched_class

            # Assign the detected class or default to "keine Angabe" (kA)
            layer.prediction_uscs_class = detected_class if detected_class else USCSClasses.kA


class BertClassifier:
    """Classifier class that uses the BERT model."""

    def __init__(self, model_path: Path):
        if model_path is None:
            # load pretrained from transformers lib (bad)
            model_path = model_config["model_path"]
        self.model_path = model_path
        self.bert_model = BertModel(model_path)

    def log_model_name_to_mlflow(self):
        if not mlflow_tracking:
            return
        mlflow.log_param("model_name", "/".join(self.model_path.parts[-2:]))

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the description of the LayerInformations objects.

        This method will populate the prediction_uscs_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object
        """
        # for unbatched 501.22s
        # for layer in layer_descriptions:
        #     layer.prediction_uscs_class = self.bert_model.predict_uscs_class(layer.material_description)

        # for batched: 386.75s
        # texts = [layer.material_description for layer in layer_descriptions]
        # predictions = self.bert_model.predict_uscs_class_batched(
        #     texts, batch_size=model_config["inference_batch_size"]
        # )
        # for layer, prediction in zip(layer_descriptions, predictions, strict=True):
        #     layer.prediction_uscs_class = prediction

        # using dummy trainer 191.745s
        eval_dataset = self.bert_model.get_tokenized_dataset(layer_descriptions)
        trainer = Trainer(
            model=self.bert_model.model,
            processing_class=self.bert_model.tokenizer,
        )
        output = trainer.predict(eval_dataset)
        predicted_indices = list(np.argmax(output.predictions, axis=1))

        # Convert indices to USCSClasses and assign them
        for layer, idx in zip(layer_descriptions, predicted_indices, strict=True):
            layer.prediction_uscs_class = self.bert_model.id2classEnum[idx]
