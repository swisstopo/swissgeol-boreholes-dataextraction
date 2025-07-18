"""Model module."""

import logging
import os

# from collections.abc import Callable
from pathlib import Path

import datasets
import torch

# from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformation

logger = logging.getLogger(__name__)


class BertModel:
    """Class for BERT model and tokenizer."""

    def __init__(self, model_path: str | Path, classification_system: type[ClassificationSystem]):
        """Initialize a pretrained BERT model from the transformers librairy.

        Args:
            model_path (str): A string containing the model name.
            classification_system (type[ClassificationSystem]): The classification system used to classify the data.
        """
        self.classification_system = classification_system
        self._setup_classification_system()

        self.model_path = model_path
        self._load_model()

    def _setup_classification_system(self) -> None:
        """Set up label mappings based on the classification system."""
        classes = self.classification_system.get_enum()
        self.num_class = len(classes)
        self.id2label = {class_.value: class_.name for class_ in classes}
        self.label2id = {class_.name: class_.value for class_ in classes}
        self.id2classEnum = {class_.value: class_ for class_ in classes}

    def _load_model(self) -> None:
        """Load the model and tokennizer with the correct number of labels and mappings."""
        # Check if model_path is a valid directory for a locally saved model
        if os.path.isdir(self.model_path):
            logger.info(f"Model and tokenizer loaded from local path: {self.model_path}")
        else:
            logger.info(f"Pretrained model and tokenizer loaded from remote source: {self.model_path}")

        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.model_path)
        try:
            self.model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_class,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except RuntimeError as e:
            error_message = str(e)
            if "size mismatch for classifier" in error_message:
                raise ValueError(
                    f"Model loading failed due to a mismatch in the number of output classes.\n"
                    f"Expected {self.num_class} classes, but the loaded model seems to have a different number.\n"
                    f"This often happens if you try to load a model trained for a different task (e.g., lithology vs"
                    f" USCS classification).\nOriginal error: {error_message}"
                ) from e
            else:
                raise  # Re-raise the original exception if it's not a size mismatch

    def freeze_all_layers(self):
        """Freeze all layers (base bert model + classifier)."""
        for name, param in self.model.named_parameters():
            logger.debug(f"Freezing Param: {name}")
            param.requires_grad = False

    def freeze_layers_except_pooler_and_classifier(self):
        """Freeze all layers except the pooler and classifier layers."""
        self.freeze_all_layers()
        self.unfreeze_pooler()
        self.unfreeze_classifier()

    def unfreeze_list(self, unfreeze_list: list[str]):
        """Unfreeze a list of layers.

        Args:
            unfreeze_list (list[str]): A list of layers to unfreeze. Possible values are:
                - "classifier"
                - "pooler"
                - "layer_11"
                - "layer_10"
                - "all"
        """
        if not unfreeze_list:
            logger.warning("No layer to unfreeze, the model will not be trained.")
        if "all" in unfreeze_list:
            logger.warning("Warning: Unfreezing all layers may consume excessive RAM and raise an error.")
            self.unfreeze_all_layers()
            return
        for layer in unfreeze_list:
            if layer == "classifier":
                self.unfreeze_classifier()
            elif layer == "pooler":
                self.unfreeze_pooler()
            elif layer == "layer_11":
                self.unfreeze_layer_11()
            elif layer == "layer_10":
                self.unfreeze_layer_10()
            else:
                raise ValueError(f"Uknown layer to unfreeze: {layer}.")

    def unfreeze_classifier(self):
        """Unfreeze all the classifier layers.

        This will put requires_grad=True for the following parameters:
            - classifier.weight
            - classifier.bias
        """
        for name, param in self.model.named_parameters():
            if name.startswith("classifier."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_pooler(self):
        """Unfreeze all the pooler layers.

        This will put requires_grad=True for the following parameters:
            - bert.pooler.dense.weight
            - bert.pooler.dense.bias
        """
        for name, param in self.model.named_parameters():
            if name.startswith("bert.pooler."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_layer_11(self):
        """Unfreeze the last layer of the transformer encoder, the 11th layer.

        The 11th layer is a basic self-attention bloc and it has all the following parameters:
            - bert.encoder.layer.11.attention.self.query.weight
            - bert.encoder.layer.11.attention.self.query.bias
            - bert.encoder.layer.11.attention.self.key.weight
            - bert.encoder.layer.11.attention.self.key.bias
            - bert.encoder.layer.11.attention.self.value.weight
            - bert.encoder.layer.11.attention.self.value.bias
            - bert.encoder.layer.11.attention.output.dense.weight
            - bert.encoder.layer.11.attention.output.dense.bias
            - bert.encoder.layer.11.attention.output.LayerNorm.weight
            - bert.encoder.layer.11.attention.output.LayerNorm.bias
            - bert.encoder.layer.11.intermediate.dense.weight
            - bert.encoder.layer.11.intermediate.dense.bias
            - bert.encoder.layer.11.output.dense.weight
            - bert.encoder.layer.11.output.dense.bias
            - bert.encoder.layer.11.output.LayerNorm.weight
            - bert.encoder.layer.11.output.LayerNorm.bias
        """
        for name, param in self.model.named_parameters():
            if name.startswith("bert.encoder.layer.11."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_layer_10(self):
        """Unfreeze the second-to-last layer of the transformer encoder, the 10th layer.

        The 10th layer is a basic self-attention bloc and it has all the following parameters:
            - bert.encoder.layer.10.attention.self.query.weight
            - bert.encoder.layer.10.attention.self.query.bias
            - bert.encoder.layer.10.attention.self.key.weight
            - bert.encoder.layer.10.attention.self.key.bias
            - bert.encoder.layer.10.attention.self.value.weight
            - bert.encoder.layer.10.attention.self.value.bias
            - bert.encoder.layer.10.attention.output.dense.weight
            - bert.encoder.layer.10.attention.output.dense.bias
            - bert.encoder.layer.10.attention.output.LayerNorm.weight
            - bert.encoder.layer.10.attention.output.LayerNorm.bias
            - bert.encoder.layer.10.intermediate.dense.weight
            - bert.encoder.layer.10.intermediate.dense.bias
            - bert.encoder.layer.10.output.dense.weight
            - bert.encoder.layer.10.output.dense.bias
            - bert.encoder.layer.10.output.LayerNorm.weight
            - bert.encoder.layer.10.output.LayerNorm.bias
        """
        for name, param in self.model.named_parameters():
            if name.startswith("bert.encoder.layer.10."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_all_layers(self):
        """Unfreeze all layers (base model + classifier)."""
        for name, param in self.model.named_parameters():
            logger.debug(f"Unfreezing Param: {name}")
            param.requires_grad = True

    def get_tokenized_dataset(self, layers: list[LayerInformation]) -> datasets.Dataset:
        """Create a tokenized datasets.Dataset object from a list of layers.

        Args:
            layers (list[LayerInformation]): A list of layers.

        Returns:
            datasets.Dataset: the dataset, with tokenized informations.
        """
        shorten = len(layers)  # used for debugging to load only a subset of data (e.g. to test overfitting the model)
        data: dict[str, list] = {
            "layer": [layer.material_description for layer in layers[:shorten]],
            "label": [layer.ground_truth_class.value for layer in layers[:shorten]],
        }
        dataset = datasets.Dataset.from_dict(data)
        return self.tokenize_dataset(dataset)

    def tokenize_dataset(self, dataset: datasets.Dataset):
        """Tokenizes the whole dataset.

        Args:
            dataset (datasets.Dataset): A dataset from the datasets librairy (transformers). Must have a column
                named "layer".

        Returns:
            datasets.Dataset: the dataset, with tokenized informations.
        """

        def tokenize(entry):
            return self.tokenize_text(entry["layer"])

        return dataset.map(tokenize, batched=True)

    def tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """Tokenizes a single text string or a list of strings.

        Args:
            text (str): The text to tokenize.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the tokenized informations. The dictionary has keys
                "input_ids", "token_type_ids" and "attention_mask"
        """
        tokenized_text = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )
        return tokenized_text

    def predict_idx(self, text: str) -> int:
        """Runs prediction on a single text input.

        Args:
            text (str): the text to predict the label index from.

        Returns:
            int: the index of the predicted label.
        """
        inputs = self.tokenize_text(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        idx = torch.argmax(outputs.logits, dim=1).item()
        return idx

    def predict_class(self, text: str) -> ClassificationSystem.EnumMember:
        """Runs prediction on a single text input.

        Args:
            text (str): the text to predict the label index from.

        Returns:
            ClassificationSystem.EnumMember: The predicted class the text input.
        """
        idx = self.predict_idx(text)
        return self.id2classEnum[idx]

    def predict_class_batched(self, texts: list[str], batch_size: int) -> list[ClassificationSystem.EnumMember]:
        """Runs batch prediction on multiple text inputs.

        Args:
            texts (list[str]): List of text descriptions to classify.
            batch_size (int): batch size for the inference.

        Returns:
            list[ClassificationSystem.EnumMember]: The predicted class for each text.
        """
        inputs = [self.tokenize_text(text) for text in texts]

        def collate_fn(batch):
            """Collates tokenized inputs into a batch-friendly format."""
            keys = batch[0].keys()
            return {key: torch.stack([b[key].squeeze(0) for b in batch]) for key in keys}

        dataloader = torch.utils.data.DataLoader(inputs, batch_size=batch_size, collate_fn=collate_fn)

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                outputs = self.model(**batch)
                predicted_indices = torch.argmax(outputs.logits, dim=1).tolist()
                predictions.extend(predicted_indices)

        # Convert indices to Enum classes
        return [self.id2classEnum[idx] for idx in predictions]
