"""Classifier module."""

import asyncio
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import boto3
import numpy as np
from description_classification.models.model import BertModel
from description_classification.utils.data_loader import LayerInformations
from description_classification.utils.data_utils import write_api_failures, write_predictions
from description_classification.utils.uscs_classes import USCSClasses, map_most_similar_uscs
from nltk.stem.snowball import SnowballStemmer
from stratigraphy.util.util import read_params
from transformers import Trainer, TrainingArguments

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

    The BaselineClassifier works by matching stemmed USCS patterns against layer descriptions using
    a flexible ordered sequence matching algorithm.
    """

    def __init__(self, match_threshold=0.75):
        """Initialize with configurable threshold.

        Args:
            match_threshold (float): Minimum coverage for matches (default: 0.75)
        """
        self.match_threshold = match_threshold

        self.uscs_patterns = classification_params["uscs_patterns"]

        self.stemmer_languages = {"de": "german", "fr": "french", "en": "english", "it": "italian"}

        self.stemmers = {}

    def get_stemmer(self, language: str) -> SnowballStemmer:
        """Get or create a stemmer for the specified language with German as a fallback option.

        Args:
            language (str): The language code for which to get the stemmer

        Returns:
            SnowballStemmer: The stemmer for the specified language
        """
        if language not in self.stemmers:
            stemmer_lang = self.stemmer_languages.get(language, "german")
            self.stemmers[language] = SnowballStemmer(stemmer_lang)

        return self.stemmers[language]

    def find_ordered_sequence(self, pattern_tokens, description_tokens, match_threshold) -> tuple | None:
        """Find the best match for pattern tokens within description tokens.

        This method searches for pattern tokens within description tokens in sequential order,
        allowing for discontinuous matches (matching allows for gaps between pattern tokens).

        Args:
            pattern_tokens (list): List of tokens to search for.
            description_tokens (list): List of tokens to search within.
            match_threshold (float): Minimum coverage ratio required for a match to be valid.

        Returns:
            tuple| None: If a match is found, returns a tuple with:
                - coverage (float): Ratio of matched pattern tokens to total pattern tokens
                - matched_positions (tuple): Positions of matches in description_tokens
                - matched_words (list): The actual matched words
        """
        if not pattern_tokens:
            return None

        description_len = len(description_tokens)
        pattern_len = len(pattern_tokens)

        # Look for partial sequence matches with flexible position matching
        matched_words = []
        last_match_pos = -1
        matched_positions = []

        for p_token in pattern_tokens:
            # Look for this pattern token anywhere after the last match
            for d_idx in range(last_match_pos + 1, description_len):
                if p_token == description_tokens[d_idx]:
                    matched_positions.append(d_idx)
                    matched_words.append(description_tokens[d_idx])
                    last_match_pos = d_idx
                    break

        coverage = len(matched_positions) / pattern_len
        if coverage >= match_threshold:
            return coverage, tuple(matched_positions), matched_words

        return None

    def classify(self, layer_descriptions: list[LayerInformations]):
        """Classifies the material descriptions of layer information objects into USCS soil classes.

        The method modifies the input object, layer_descriptions by setting their prediction_uscs_class attribute.
        The approach is as follows:

        1. Tokenize and stem both the material description and the USCS pattern keywords
        2. Find matches between description and patterns using partial matching
        3. Scores matches based on three criteria (in priority order):
           - Coverage: Percentage of pattern words matched in the description
           - Complexity: Length/specificity of the pattern (longer patterns preferred)
           - Position: Earlier matches in the text are preferred
        4. Assigns the best matching USCS class to the layer object

        For layers with no matches, assigns the default class 'kA' (no classification).

        Args:
            layer_descriptions (list[LayerInformations]): The LayerInformations object
        """
        for layer in layer_descriptions:
            patterns = self.uscs_patterns[layer.language]
            description = layer.material_description.lower()
            language = layer.language
            stemmer = self.get_stemmer(language)

            # Tokenize the description into separate words and stem them
            description_tokens = re.findall(r"\b\w+\b", description.lower())
            stemmed_description_tokens = [stemmer.stem(token) for token in description_tokens]

            matches = []

            for class_key, class_keyphrases in patterns.items():
                uscs_class = map_most_similar_uscs(class_key)
                for class_keyphrase in class_keyphrases:
                    # Tokenize the pattern into separate words and stem them
                    pattern_tokens = re.findall(r"\b\w+\b", class_keyphrase)
                    stemmed_pattern_tokens = [stemmer.stem(token) for token in pattern_tokens]

                    result = self.find_ordered_sequence(
                        stemmed_pattern_tokens,
                        stemmed_description_tokens,
                        self.match_threshold,
                    )

                    if result:
                        coverage, match_positions, matched_words = result
                        matches.append(
                            {
                                "class": uscs_class,
                                "coverage": coverage,
                                "complexity": len(pattern_tokens),
                                "matched_words": matched_words,
                                "match_positions": match_positions,
                            }
                        )

            # Sort matches by coverage and complexity in descending order, then by match_positions in ascending order
            sorted_matches = sorted(matches, key=lambda x: (-x["coverage"], -x["complexity"], x["match_positions"]))

            if sorted_matches:
                layer.prediction_uscs_class = sorted_matches[0]["class"]
            else:
                layer.prediction_uscs_class = USCSClasses.kA


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
            args=TrainingArguments(per_device_eval_batch_size=model_config["inference_batch_size"]),
        )
        output = trainer.predict(eval_dataset)
        predicted_indices = list(np.argmax(output.predictions, axis=1))

        # Convert indices to USCSClasses and assign them
        for layer, idx in zip(layer_descriptions, predicted_indices, strict=True):
            layer.prediction_uscs_class = self.bert_model.id2classEnum[idx]


class AWSBedrockClassifier:
    """AWSBedrockClassifier class uses AWS Bedrock with underlying Anthropic LLM models."""

    def __init__(
        self,
        bedrock_out_directory: Path | None,
        max_tokens: int = 256,
        temperature: float = 0.3,
        max_concurrent_calls: int = 1,
    ):
        """Creates a boto3 client for AWS Bedrock and initializes the classifier.

        Environment variables are used to configure the AWS region, model ID, and Anthropic version.
        The USCS patterns and classification prompts are read from the configuration files.

        Args:
            max_tokens (int): The maximum number of tokens to generate.
            bedrock_out_directory (Path): Directory to write prediction outputs and API failures
            temperature (float): The sampling temperature to use.
            store_files (bool): Whether to store the prediction files (default: False)
            max_concurrent_calls (int): Maximum number of concurrent API calls (default: 1)
        """
        self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION"))
        self.anthropic_version = os.environ.get("ANTHROPIC_VERSION")
        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID")

        self.uscs_patterns = classification_params["uscs_patterns"]

        self.bedrock_out_directory = bedrock_out_directory
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent_calls = max_concurrent_calls

    def create_message(
        self, max_tokens: int, temperature: float, anthropic_version: str, material_description: str, language: str
    ) -> dict:
        """Creates a message for the Anthropic LLM model.

        Args:
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature to use.
            anthropic_version (str): The version of the Anthropic model to use.
            material_description (str): The material description to classify.
            language (str): The language of the material description.

        Returns:
            body (dict): The message body for the Bedrock API.
        """
        language_patterns = self.uscs_patterns[language]

        classification_prompts = read_params(os.environ.get("ANTHROPIC_PROMPT_TEMPLATE"))
        system_message = classification_prompts["system_prompt"].format(uscs_patterns=language_patterns)
        user_message = classification_prompts["user_prompt"].format(material_description=material_description)

        body = json.dumps(
            {
                "anthropic_version": anthropic_version,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_message,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message,
                            }
                        ],
                    },
                ],
            }
        )

        return body

    def format_response(self, response: dict) -> dict:
        """Formats the response from the Bedrock API.

        The function extracts the model answer (USCS Class) and reasoning from the llm response object.
        If no reasoning is provided None is assigned to the Reasoning key, if no answer is provided
        default value 'kA' is assigned to the Model Answer key.

        Args:
            response (dict): The response from the Bedrock API.

        Returns:
            dict: A dictionary containing the model answer (USCS Class) and reasoning.
        """
        response_body = json.loads(response.get("body").read())
        response_text = response_body.get("content")[0].get("text", None)

        reasoning = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
        reasoning = reasoning.group(1).strip() if reasoning else None

        answer = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        answer = answer.group(1).strip() if answer else "kA"

        return {"Model Answer": answer, "Reasoning": reasoning}

    async def classify_async(self, layer_descriptions: list[LayerInformations]):
        """Classifies the material descriptions of layer information objects into USCS soil classes.

        The method modifies the input object, layer_descriptions by setting their bprediction_uscs_class attribute.
        The approach is as follows:
        1. Each layer description together with the detected language is added to the prompt sent to an Anthropic
        LLM model API on AWS Bedrock.
        2. The LLM model provides an answer in the form of a USCS class and (potentially) reasoning.
        3. If the USCS class and Reasoning exists in the LLM response both are added to the layer_descriptions object.
        """
        api_failures = []
        run_id = str(uuid.uuid4())

        layers_by_filename = defaultdict(list)
        for layer in layer_descriptions:
            layers_by_filename[layer.filename].append(layer)

        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        for filename, filename_layers in layers_by_filename.items():
            print(f"Processing file: {filename} with {len(filename_layers)} layers")
            path = f"{Path(filename).stem}.csv"

            async def process_layer(layer):
                async with semaphore:
                    try:
                        print(f"Classifying layer: {layer.filename}_{layer.borehole_index}_{layer.layer_index}")

                        body = self.create_message(
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            anthropic_version=self.anthropic_version,
                            material_description=layer.material_description,
                            language=layer.language,
                        )

                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None,  # Use default executor
                            lambda: self.bedrock_client.invoke_model(body=body, modelId=self.model_id),
                        )

                        formatted_response = self.format_response(response)
                        uscs_class = map_most_similar_uscs(formatted_response.get("Model Answer"))

                        layer.prediction_uscs_class = uscs_class
                        layer.llm_reasoning = formatted_response.get("Reasoning")

                        return None

                    except Exception as e:
                        error_msg = str(e)
                        print(f"API call failed for '{layer.filename}, {layer.layer_index}': {error_msg}")
                        layer.prediction_uscs_class = USCSClasses.kA

                        # Return failure info
                        return {
                            "run_id": run_id,
                            "filename": layer.filename,
                            "borehole_index": layer.borehole_index,
                            "layer_index": layer.layer_index,
                            "error": error_msg,
                        }

            tasks = [process_layer(layer) for layer in filename_layers]
            results = await asyncio.gather(*tasks)

            for result in results:
                if result is not None:
                    api_failures.append(result)

            if self.bedrock_out_directory:
                write_predictions(filename_layers, self.bedrock_out_directory, path)
                write_api_failures(api_failures, self.bedrock_out_directory)

    def classify(self, layer_descriptions: list[LayerInformations]):
        asyncio.run(self.classify_async(layer_descriptions))
