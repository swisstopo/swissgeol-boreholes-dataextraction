"""Bedrock LLM-based classifier module."""

import asyncio
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path

import boto3
from classification.utils.classification_classes import USCSSystem
from classification.utils.data_loader import LayerInformations
from classification.utils.data_utils import write_api_failures, write_predictions
from utils.file_utils import read_params


class AWSBedrockClassifier:
    """AWSBedrockClassifier class uses AWS Bedrock with underlying Anthropic LLM models."""

    def __init__(self, bedrock_out_directory: Path | None, max_concurrent_calls: int = 1, api_call_delay: float = 0.0):
        """Creates a boto3 client for AWS Bedrock and initializes the classifier.

        Environment variables are used to configure the AWS region, model ID, and Anthropic version.
        The USCS patterns and classification prompts are read from the configuration files.

        Args:
            bedrock_out_directory (Path): Directory to write prediction outputs and API failures
            max_concurrent_calls (int): Maximum number of concurrent API calls (default: 1)
            api_call_delay (float): Delay between API calls in seconds (default: 0)
        """
        self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION"))
        self.anthropic_version = os.environ.get("ANTHROPIC_VERSION")
        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID")

        # Bedrock parameters
        self.config = read_params("bedrock/bedrock_config.yml")
        reasoning_mode = self.config["reasoning_mode"]

        self.uscs_patterns = read_params("bedrock/classification_patterns_bedrock.yml")["uscs_patterns"][
            self.config["uscs_pattern_version"]
        ]

        if reasoning_mode:
            self.classification_prompts = read_params("bedrock/classification_prompts.yml")["reasoning"][
                self.config["prompt_version"]
            ]
        else:
            self.classification_prompts = read_params("bedrock/classification_prompts.yml")["classification"][
                self.config["prompt_version"]
            ]

        self.bedrock_out_directory = bedrock_out_directory
        self.max_concurrent_calls = max_concurrent_calls
        self.api_call_delay = api_call_delay

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

        system_message = self.classification_prompts["system_prompt"].format(uscs_patterns=language_patterns)
        user_message = self.classification_prompts["user_prompt"].format(material_description=material_description)

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

        layers_by_filename: dict[str, list[LayerInformations]] = defaultdict(list)
        for layer in layer_descriptions:
            layers_by_filename[layer.filename].append(layer)

        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        for filename, filename_layers in layers_by_filename.items():
            print(f"Processing file: {filename} with {len(filename_layers)} layers")
            path = f"{Path(filename).stem}.json"

            async def process_layer(layer: LayerInformations):
                async with semaphore:
                    try:
                        print(f"Classifying layer: {layer.filename}_{layer.borehole_index}_{layer.layer_index}")

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
                        uscs_class = USCSSystem.map_most_similar_class(formatted_response.get("Model Answer"))

                        layer.prediction_class = uscs_class
                        layer.llm_reasoning = formatted_response.get("Reasoning")

                        return None

                    except Exception as e:
                        error_msg = str(e)
                        print(f"API call failed for '{layer.filename}, {layer.layer_index}': {error_msg}")
                        layer.prediction_class = USCSSystem.get_default_class_value()

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
