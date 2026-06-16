"""Bedrock LLM-based classifier module."""

import asyncio
import itertools
import json
import logging
import os
from pathlib import Path

import anthropic
import backoff
import mlflow
import pydantic_core
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from classification.classifiers.classifier import Classifier
from classification.utils.data_utils import read_predictions, write_predictions
from classification.utils.datasets.classification import ClassificationSystem, LayerInformation
from classification.utils.file_utils import read_params

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AWSBedrockEntry(BaseModel):
    """A single classification result returned by the Bedrock LLM for one layer.

    Attributes:
        index (int): Position of the layer in the batch sent to the model, used to align predictions back to inputs.
        class_ (str | None): Predicted class label for single-label classification.
        classes_ (list[str] | None): Predicted class labels for multi-label classification.
        reasoning (str): Reasoning of the predicted output.

    """

    index: int
    class_: str | None = None
    classes_: list[str] | None = None
    reasoning: str | None = None


class AWSBedrockPrediction(BaseModel):
    """Structured output parsed from the Bedrock tool-use response for a batch of layers.

    Attributes:
        predictions: Ordered list of per-layer predictions. Must have the same length as the input batch.
    """

    predictions: list[AWSBedrockEntry]


class AWSBedrockClassifier(Classifier):
    """Classifier that uses AWS Bedrock to call Anthropic Claude models for layer classification.

    Sends batched layer descriptions to the model via the Anthropic Bedrock client using
    structured tool-use output. Supports optional chain-of-thought reasoning mode.
    Classification patterns and prompts are loaded from versioned configuration files.
    """

    def __init__(
        self,
        bedrock_out_directory: Path | None,
        classification_system: type[ClassificationSystem],
        max_concurrent_calls: int = 3,
        use_local_cache: bool = False,
    ):
        """Creates a boto3 client for AWS Bedrock and initializes the classifier.

        Environment variables are used to configure the AWS region, model ID, and Anthropic version.
        The class patterns and classification prompts are read from the configuration files.

        Args:
            bedrock_out_directory (Path | None): Directory to write prediction outputs and API failures
            classification_system (type[ClassificationSystem]): the classification system used
            max_concurrent_calls (int): Max number of concurrent calls. Defaults to 3.
            use_local_cache (bool): Enable local file caching to avoid costs of reprocessing
                files. Default to False.
        """
        self.init_config(classification_system)
        self.classification_system = classification_system
        self.bedrock_out_directory = bedrock_out_directory
        self.max_concurrent_calls = max_concurrent_calls
        self.use_local_cache = use_local_cache

        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID")
        self.model_region = os.environ.get("AWS_DEFAULT_REGION")
        self.pattern_version = self.config["pattern_version"]
        self.prompt_version = self.config["prompt_version"]
        self.reasoning_mode = self.config["reasoning_mode"]
        self.multilabel_mode = self.config.get("multilabel_mode", False)

        # Async functions
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        self.bedrock_client = anthropic.AsyncAnthropicBedrock(aws_region=self.model_region)

        # Load classification instructions
        self.class_examples = read_params(self.config["pattern_file"])[self.pattern_version]

        # Load tool and system prompt
        prompt_section = "reasoning" if self.reasoning_mode else "classification"
        self.system_prompts = read_params(self.config["prompts_file"])[prompt_section][self.prompt_version]

        if self.multilabel_mode:
            tool_key = "multilabel"
        elif self.reasoning_mode:
            tool_key = "reasoning"
        else:
            tool_key = "classification"
        self.tool = read_params(self.config["tool_file"])[tool_key]

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "bedrock"

    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.APIStatusError, ValueError, pydantic_core.ValidationError),
        max_tries=5,
    )
    async def _call_bedrock(self, filename_layers: list[LayerInformation]) -> list[AWSBedrockEntry]:
        """Call the Bedrock API, retrying on transient errors.

        Args:
            filename_layers: List of layers sent to the model.

        Returns:
            Parsed list of per-layer predictions aligned to the input.

        Raises:
            pydantic_core.ValidationError: Re-raised after all retries are exhausted.
            ValueError: Re-raised after all retries are exhausted.
            anthropic.RateLimitError: Re-raised after all retries are exhausted.
            anthropic.APIStatusError: Re-raised after all retries are exhausted.
        """
        message = await self.bedrock_client.messages.create(
            model=self.model_id,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            tools=[
                {
                    **self.tool,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tool_choice={"type": "tool", "name": self.tool["name"]},
            system=[
                {
                    "type": "text",
                    "text": self.system_prompts.format(class_patterns=self.class_examples),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": json.dumps({i: t.material_description for i, t in enumerate(filename_layers)}),
                }
            ],
        )
        tool_result = next(b for b in message.content if b.type == "tool_use")
        predictions = AWSBedrockPrediction.model_validate(tool_result.input).predictions

        if len(predictions) != len(filename_layers):
            raise ValueError(f"Wrong number of prediction {len(filename_layers)=}, {len(predictions)=}")

        return predictions

    def log_params(self):
        """Log model and id, prompt and parameter versions if anthropic model used."""
        mlflow.log_param("anthropic_model_id", self.model_id)
        mlflow.log_param("anthropic_prompt_version", self.prompt_version)
        mlflow.log_param("anthropic_class_pattern_version", self.pattern_version)
        mlflow.log_param("anthropic_reasoning_mode", self.reasoning_mode)

    async def _classify_file(self, filename: str, filename_layers: list[LayerInformation]) -> list[LayerInformation]:
        """Classify all layers belonging to a single borehole file in one batched API call.

        Sends all layer material descriptions as a numbered list to the model and parses the
        structured tool-use response back into per-layer predictions. On API failure, every layer
        in the batch is assigned the default class for the classification system. Writes results
        to disk when ``bedrock_out_directory`` is set.

        Args:
            filename: Source filename used as the batch identifier and output stem.
            filename_layers: Ordered list of layers from that file whose ``prediction_class``
                and ``llm_reasoning`` fields are updated in-place.

        Returns:
            list[LayerInformation]: Classified layer information.
        """
        output_path = (
            (self.bedrock_out_directory / f"{Path(filename).stem}.json") if self.bedrock_out_directory else None
        )

        if self.use_local_cache and output_path and output_path.exists():
            return read_predictions(output_path, self.classification_system)

        async with self.semaphore:
            try:
                predictions = await self._call_bedrock(filename_layers)
            except Exception as e:
                logger.warning(f"API call failed for '{filename}': {str(e)}")
                default = self.classification_system.get_default_class_value().name
                if self.multilabel_mode:
                    predictions = [AWSBedrockEntry(index=i, classes_=[default]) for i, _ in enumerate(filename_layers)]
                else:
                    predictions = [AWSBedrockEntry(index=i, class_=default) for i, _ in enumerate(filename_layers)]

        # Update predictions (label and reasoning)
        for data in predictions:
            filename_layers[data.index].prediction_class = self.classification_system.map_most_similar_class(
                data.class_
            )
            filename_layers[data.index].llm_reasoning = data.reasoning

        if self.bedrock_out_directory:
            write_predictions(filename_layers, str(output_path))

        return filename_layers

    async def classify_async(self, layer_descriptions: list[LayerInformation]) -> list[LayerInformation]:
        """Classify all layers asynchronously, grouped by source file.

        Layers are sorted and grouped by filename so each borehole file is processed in a single
        batched API call. All file-level tasks are awaited concurrently via ``asyncio.gather``.

        Args:
            layer_descriptions: All layers to classify, potentially spanning multiple files.

        Returns:
            list[LayerInformation]: Updated layer information
        """
        # Sort layers for grouping
        layer_descriptions = sorted(layer_descriptions, key=lambda layer: layer.filename)

        tasks = [
            self._classify_file(filename, list(layers))
            for filename, layers in itertools.groupby(layer_descriptions, key=lambda layer: layer.filename)
        ]
        return await tqdm_asyncio.gather(*tasks, desc="Classifying files")

    def classify(self, layer_descriptions: list[LayerInformation]) -> list[LayerInformation]:
        """Classify all layers using the Bedrock API.

        Synchronous entry point that runs the async classification pipeline via
        ``asyncio.run``. Layers are grouped by source file and processed concurrently.

        Args:
            layer_descriptions (list[LayerInformation]): All layers to classify, potentially spanning multiple files.

        Returns:
            list[LayerInformation]: Classified layer information.
        """
        return [layer for file_layers in asyncio.run(self.classify_async(layer_descriptions)) for layer in file_layers]
