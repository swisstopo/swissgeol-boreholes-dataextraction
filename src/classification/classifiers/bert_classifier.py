"""Bert-based classifier module."""

from pathlib import Path

import mlflow

from classification.classifiers.classifier import Classifier
from classification.models.model import BertModel
from classification.utils.datasets.classification import ClassificationSystem, LayerInformation


class BertClassifier(Classifier):
    """Classifier class that uses the BERT model."""

    def __init__(
        self,
        model_path: Path | str | None,
        classification_system: type[ClassificationSystem],
        backbone_path: Path | None = None,
        tokenizer_path: Path | None = None,
    ):
        """Initialize a BertClassifier instance.

        Args:
            model_path (Path | str | None): Local path to the model directory or a HuggingFace model ID string.
                For split models this is the head directory; for full models it is the complete HuggingFace
                model directory or repo ID.
            classification_system (type[ClassificationSystem]): the classification system used to classify
                the descriptions.
            backbone_path (Path | None): Path to backbone.safetensors for split-model loading.
                When provided, model_path is treated as the head directory.
            tokenizer_path (Path | None): Directory containing the tokenizer files. When None, falls back
                to model_path.
        """
        self.init_config(classification_system)
        if model_path is None:
            # load pretrained from transformers lib (bad)
            model_path = Path(self.config["model_path"])
        self.model_path: Path | str = model_path
        self.bert_model = BertModel(
            model_path, classification_system, backbone_path=backbone_path, tokenizer_path=tokenizer_path
        )

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "bert"

    def log_params(self):
        """Log the name of the model used."""
        model_name = str(self.model_path)
        if isinstance(self.model_path, Path):
            model_name = "/".join(self.model_path.parts[-2:])
        mlflow.log_param("model_name", model_name)

    def classify(self, layer_descriptions: list[LayerInformation]) -> list[LayerInformation]:
        """Classifies the description of the LayerInformation objects.

        This method will populate the prediction_class attribute of each object.

        Args:
            layer_descriptions (list[LayerInformation]): The LayerInformation object

        Returns:
            list[LayerInformation]: The updated LayerInformation object
        """
        predictions = self.bert_model.predict_class_batched(
            [layer.material_description for layer in layer_descriptions],
            batch_size=self.config["inference_batch_size"],
        )

        # Convert indices to Enum classes and assign them
        for layer, prediction_idx in zip(layer_descriptions, predictions, strict=True):
            layer.prediction_class = prediction_idx

        return layer_descriptions
