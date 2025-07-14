"""Bedrock LLM-based classifier module."""

import asyncio
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path

import boto3
import mlflow
from tqdm import tqdm

from classification.classifiers.classifier import Classifier
from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformation
from classification.utils.data_utils import write_api_failures, write_predictions
from utils.file_utils import read_params

logger = logging.getLogger(__name__)


class AWSBedrockClassifier(Classifier):
    """AWSBedrockClassifier class uses AWS Bedrock with underlying Anthropic LLM models."""

    def __init__(
        self,
        bedrock_out_directory: Path | None,
        classification_system: type[ClassificationSystem],
        max_concurrent_calls: int = 1,
        api_call_delay: float = 0.0,
    ):
        """Creates a boto3 client for AWS Bedrock and initializes the classifier.

        Environment variables are used to configure the AWS region, model ID, and Anthropic version.
        The class patterns and classification prompts are read from the configuration files.

        Args:
            bedrock_out_directory (Path): Directory to write prediction outputs and API failures
            classification_system (type[ClassificationSystem]): the classification system used
            max_concurrent_calls (int): Maximum number of concurrent API calls (default: 1)
            api_call_delay (float): Delay between API calls in seconds (default: 0)
        """
        self.init_config(classification_system)
        self.classification_system = classification_system
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION")
        )
        self.anthropic_version = os.environ.get("ANTHROPIC_VERSION")
        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID")

        # Bedrock parameters
        reasoning_mode = self.config["reasoning_mode"]

        self.pattern_version = self.config["pattern_version"]
        self.class_patterns = read_params(self.config["pattern_file"])[self.pattern_version]

        self.reasoning_mode = reasoning_mode
        str_mode = "reasoning" if self.reasoning_mode else "classification"
        self.prompt_version = self.config["prompt_version"]
        self.classification_prompts = read_params(self.config["prompts_file"])[str_mode][self.prompt_version]

        self.bedrock_out_directory = bedrock_out_directory
        self.max_concurrent_calls = max_concurrent_calls
        self.api_call_delay = api_call_delay

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "bedrock"

    def log_params(self):
        """Log model and id, prompt and parameter versions if anthropic model used."""
        mlflow.log_param("anthropic_model_id", os.environ.get("ANTHROPIC_MODEL_ID"))
        mlflow.log_param("anthropic_prompt_version", self.prompt_version)
        mlflow.log_param("anthropic_class_pattern_version", self.pattern_version)
        mlflow.log_param("anthropic_reasoning_mode", self.reasoning_mode)

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
        language_patterns = self.class_patterns

        system_message = self.classification_prompts["system_prompt"].format(class_patterns=language_patterns)
        user_message_instructions = self.classification_prompts["user_prompt_instruction"]
        user_message_description = self.classification_prompts["user_prompt_description"].format(
            material_description=material_description
        )

        # Cache control enables prompt, it works by adding portions of the prompt context to a cache, we can leverage
        # the cache to skip recomputation of inputs for prompts that are repeated, e.g., USCS or Lithology classes:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        body = json.dumps(
            {
                "anthropic_version": anthropic_version,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": [{"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message_instructions,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": user_message_description},
                        ],
                    },
                ],
            }
        )

        return body

    def format_response(self, response: dict) -> dict:
        """Formats the response from the Bedrock API.

        The function extracts the model answer (Class) and reasoning from the llm response object.
        If no reasoning is provided None is assigned to the Reasoning key, if no answer is provided
        default value 'kA' is assigned to the Model Answer key.

        Args:
            response (dict): The response from the Bedrock API.

        Returns:
            dict: A dictionary containing the model answer (Class) and reasoning.
        """
        response_body = json.loads(response.get("body").read())
        response_text = response_body.get("content")[0].get("text", None)

        reasoning = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
        reasoning = reasoning.group(1).strip() if reasoning else None

        answer = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        answer = answer.group(1).strip() if answer else "kA"

        return {"Model Answer": answer, "Reasoning": reasoning}

    async def classify_async(self, layer_descriptions: list[LayerInformation]):
        """Classifies the material descriptions of layer information objects into the chosen classification system.

        The method modifies the input object, layer_descriptions by setting their prediction_class attribute.
        The approach is as follows:
        1. Each layer description together with the detected language is added to the prompt sent to an Anthropic
        LLM model API on AWS Bedrock.
        2. The LLM model provides an answer in the form of a class and (potentially) reasoning.
        3. If the class and Reasoning exists in the LLM response both are added to the layer_descriptions object.
        """
        api_failures = []
        run_id = str(uuid.uuid4())

        layers_by_filename: dict[str, list[LayerInformation]] = defaultdict(list)
        for layer in layer_descriptions:
            layers_by_filename[layer.filename].append(layer)

        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        for filename, filename_layers in tqdm(layers_by_filename.items()):
            logger.info(f"Processing file: {filename} with {len(filename_layers)} layers")
            path = f"{Path(filename).stem}.json"

            async def process_layer(layer: LayerInformation):
                async with semaphore:
                    try:
                        logger.debug(f"Classifying layer: {layer.filename}_{layer.borehole_index}_{layer.layer_index}")

                        await asyncio.sleep(self.api_call_delay)

                        body = self.create_message(
                            max_tokens=self.config["max_tokens"],
                            temperature=self.config["temperature"],
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
                        layer.prediction_class = self.classification_system.map_most_similar_class(
                            formatted_response.get("Model Answer")
                        )
                        layer.llm_reasoning = formatted_response.get("Reasoning")

                        return None

                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(f"API call failed for '{layer.filename}, {layer.layer_index}': {error_msg}")
                        layer.prediction_class = self.classification_system.get_default_class_value()

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

    def classify(self, layer_descriptions: list[LayerInformation]):
        asyncio.run(self.classify_async(layer_descriptions))
