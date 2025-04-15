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

from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.uscs_classes import USCSClasses

logger = logging.getLogger(__name__)


class BertModel:
    """Class for BERT model and tokenizer."""

    def __init__(self, model_path: str | Path):
        """Initialize a pretrained BERT model from the transformers librairy.

        Args:
            model_path (str): A string containing the model name.
        """
        self.num_class = len(USCSClasses)
        self.id2label = {class_.value: class_.name for class_ in USCSClasses}
        self.label2id = {class_.name: class_.value for class_ in USCSClasses}
        # Check if model_path is a valid directory for a locally saved model
        if os.path.isdir(model_path):
            logger.info(f"Model and tokenizer loaded from local path: {model_path}")
        else:
            logger.info(f"Pretrained model and tokenizer loaded from remote source: {model_path}")

        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_path)
        self.model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_class,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.id2classEnum = {class_.value: class_ for class_ in USCSClasses}

    def freeze_all_layers(self):
        """Freeze all layers (base model + classifier)."""
        for name, param in self.model.named_parameters():
            logger.debug(f"Freezing Param: {name}")
            param.requires_grad = False

    def freeze_layers_except_pooler_and_classifier(self):
        """Freeze all layers except the pooler and classifier layers."""
        self.freeze_all_layers()
        self.unfreeze_pooler()
        self.unfreeze_classifier()

    def unfreeze_pooler(self):
        """Unfreeze all the pooler layers."""
        # Unfreeze pooler layer
        for name, param in self.model.named_parameters():
            if "pooler" in name:
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_classifier(self):
        """Unfreeze all the classifier layers."""
        # Unfreeze classifier layer
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_layer_11(self):
        """Unfreeze the last layer of the transformer encoder, the 11th layer.

        This concerns all the folloewing layers:
            - TODO
        """
        for name, param in self.model.bert.encoder.layer[11].named_parameters():
            logger.debug(f"Unfreezing Param: {name}")
            param.requires_grad = True

    def unfreeze_all_layers(self):
        """Unfreeze all layers (base model + classifier)."""
        for name, param in self.model.named_parameters():
            logger.debug(f"Unfreezing Param: {name}")
            param.requires_grad = True

    def get_tokenized_dataset(self, layers: list[LayerInformations]) -> datasets.Dataset:
        """Create a tokenized datasets.Dataset object from a list of layers.

        Args:
            layers (list[LayerInformations]): A list of layers.

        Returns:
            datasets.Dataset: the dataset, with tokenized informations.
        """
        shorten = len(layers)  # used for debugging to load only a subset of data (e.g. to test overfitting the model)
        data: dict[str, list] = {
            "layer": [layer.material_description for layer in layers[:shorten]],
            "label": [layer.ground_truth_uscs_class.value for layer in layers[:shorten]],
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

    def predict_uscs_class(self, text: str) -> USCSClasses:
        """Runs prediction on a single text input.

        Args:
            text (str): the text to predict the label index from.

        Returns:
            int: the uscs class of the predicted label.
        """
        idx = self.predict_idx(text)
        return self.id2classEnum[idx]

    def predict_uscs_class_batched(self, texts: list[str], batch_size: int) -> list[USCSClasses]:
        """Runs batch prediction on multiple text inputs.

        Args:
            texts (list[str]): List of text descriptions to classify.
            batch_size (int): batch size for the inference.

        Returns:
            list[USCSClasses]: Predicted USCS class for each text.
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

        # Convert indices to USCSClasses
        return [self.id2classEnum[idx] for idx in predictions]
