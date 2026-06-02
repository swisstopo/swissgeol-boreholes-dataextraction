"""Bedrock LLM-based classifier module."""

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path

import anthropic
import mlflow
from pydantic import BaseModel
from tqdm import tqdm

from classification.classifiers.classifier import Classifier
from classification.utils.data_utils import write_predictions
from classification.utils.datasets.classification import ClassificationSystem, LayerInformation
from classification.utils.file_utils import read_params

logger = logging.getLogger(__name__)


class AWSBedrockEntry(BaseModel):
    """A single classification result returned by the Bedrock LLM for one layer.

    Attributes:
        index: Position of the layer in the batch sent to the model, used to align predictions back to inputs.
        class_: Predicted class label as returned by the model.
    """

    index: int
    class_: str


class AWSBedrockPrediction(BaseModel):
    """Structured output parsed from the Bedrock tool-use response for a batch of layers.

    Attributes:
        predictions: Ordered list of per-layer predictions. Must have the same length as the input batch.
    """

    predictions: list[AWSBedrockEntry]


class AWSBedrockClassifier(Classifier):
    """AWSBedrockClassifier class uses AWS Bedrock with underlying Anthropic LLM models."""

    def __init__(self, bedrock_out_directory: Path | None, classification_system: type[ClassificationSystem]):
        """Creates a boto3 client for AWS Bedrock and initializes the classifier.

        Environment variables are used to configure the AWS region, model ID, and Anthropic version.
        The class patterns and classification prompts are read from the configuration files.

        Args:
            bedrock_out_directory (Path): Directory to write prediction outputs and API failures
            classification_system (type[ClassificationSystem]): the classification system used
        """
        self.init_config(classification_system)
        self.classification_system = classification_system
        self.bedrock_out_directory = bedrock_out_directory
        self.bedrock_client = anthropic.AnthropicBedrock(aws_region=os.environ.get("AWS_DEFAULT_REGION"))

        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID")
        self.pattern_version = self.config["pattern_version"]
        self.prompt_version = self.config["prompt_version"]
        # TODO: add reasoning logic
        self.reasoning_mode = self.config["reasoning_mode"]

        # Load classification instructions
        self.class_examples = read_params(self.config["pattern_file"])[self.pattern_version]

        # Load tool and system prompt
        prompts = read_params(self.config["prompts_file"])["reasoning" if self.reasoning_mode else "classification"][
            self.prompt_version
        ]
        self.system_prompts = prompts["system_prompt"]
        self.tool = prompts["tool"]

    def get_name(self) -> str:
        """Returns a string with the name of the classifier."""
        return "bedrock"

    def log_params(self):
        """Log model and id, prompt and parameter versions if anthropic model used."""
        mlflow.log_param("anthropic_model_id", os.environ.get("ANTHROPIC_MODEL_ID"))
        mlflow.log_param("anthropic_prompt_version", self.prompt_version)
        mlflow.log_param("anthropic_class_pattern_version", self.pattern_version)
        mlflow.log_param("anthropic_reasoning_mode", self.reasoning_mode)

    async def classify_async(self, layer_descriptions: list[LayerInformation]):
        """Classifies the material descriptions of layer information objects into the chosen classification system.

        TODO: update documentation
        The method modifies the input object, layer_descriptions by setting their prediction_class attribute.
        The approach is as follows:
        1. Each layer description together with the detected language is added to the prompt sent to an Anthropic
        LLM model API on AWS Bedrock.
        2. The LLM model provides an answer in the form of a class and (potentially) reasoning.
        3. If the class and Reasoning exists in the LLM response both are added to the layer_descriptions object.
        """
        layers_by_filename: dict[str, list[LayerInformation]] = defaultdict(list)
        for layer in layer_descriptions:
            layers_by_filename[layer.filename].append(layer)

        # TODO: add (back) concurent calls
        # TODO: add (back) retries
        for filename, filename_layers in tqdm(layers_by_filename.items()):
            logger.info(f"Processing file: {filename} with {len(filename_layers)} layers")
            predictions: list[AWSBedrockEntry] = []

            try:
                message = self.bedrock_client.messages.create(
                    model=self.model_id,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    tools=[self.tool],
                    tool_choice={"type": "tool", "name": self.tool["name"]},
                    system=self.system_prompts.format(class_patterns=self.class_examples),
                    messages=[
                        {
                            "role": "user",
                            "content": "\n".join(
                                f"{i}. {t.material_description}" for i, t in enumerate(filename_layers)
                            ),
                        },
                    ],
                )
                tool_result = next(b for b in message.content if b.type == "tool_use")
                predictions = AWSBedrockPrediction.model_validate(tool_result.input).predictions

                if len(predictions) != len(filename_layers):
                    raise ValueError(f"Wrong number of prediction {len(filename_layers)=}, {len(predictions)=}")

            except Exception as e:
                logger.warning(f"API call failed for '{filename}': {str(e)}")
                predictions = [
                    AWSBedrockEntry(index=i, class_=self.classification_system.get_default_class_value())
                    for i, _ in enumerate(filename_layers)
                ]
            # Update predictions
            for data in predictions:
                filename_layers[data.index].prediction_class = self.classification_system.map_most_similar_class(
                    data.class_
                )

            if self.bedrock_out_directory:
                write_predictions(filename_layers, self.bedrock_out_directory, f"{Path(filename).stem}.json")

    def classify(self, layer_descriptions: list[LayerInformation]):
        asyncio.run(self.classify_async(layer_descriptions))
