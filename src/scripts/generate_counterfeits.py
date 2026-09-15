"""Generate counterfeit (class-flipped) material descriptions for a classification system via Bedrock."""

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import backoff
import numpy as np
import pydantic_core
from tqdm.asyncio import tqdm_asyncio

from classification.utils.datasets import ExistingClassificationSystems
from classification.utils.datasets.classification import (
    ClassificationSystem,
    GroundTruthBoreholeWithLanguage,
    LayerInformation,
)
from classification.utils.file_utils import read_params
from core.ground_truth import (
    GroundTruth,
    GroundTruthBorehole,
    GroundTruthConsolidated,
    GroundTruthLayer,
    GroundTruthLayerDepth,
    GroundTruthMetadata,
    GroundTruthUnconsolidated,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerInformationCounterfeits:
    """A single ground-truth description paired with its Bedrock-generated counterfeit."""

    ground_truth_text: str | None
    counterfeit_text: str | None
    ground_truth_class: str | None
    counterfeit_class: str | None
    filename: str = ""
    borehole_index: int = 0
    layer_index: int = 0


class AWSBedrockCounterfeits:
    """Generates counterfeit material descriptions by asking Bedrock to rewrite them into a target class."""

    def __init__(self, examples_path: Path, max_concurrent_calls: int = 3) -> None:
        """Create the Bedrock client and load the tool/prompt/example configuration.

        Args:
            examples_path (Path): Path to the YAML file of classification examples shown to Bedrock.
            max_concurrent_calls (int): Max number of concurrent Bedrock calls. Defaults to 3.
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.model_id: str | None = os.environ.get("ANTHROPIC_MODEL_ID")
        self.model_region: str | None = os.environ.get("AWS_DEFAULT_REGION")
        self.bedrock_client = anthropic.AsyncAnthropicBedrock(aws_region=self.model_region)
        self.tool: dict = read_params("bedrock/tool/tool_generate_counterfeits.yml")
        self.class_examples: str = read_params(examples_path)["baseline"]
        self.system_prompts: str = read_params("bedrock/prompts/bedrock_generate_counterfeits_prompt.yml")

    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.APIStatusError, ValueError, pydantic_core.ValidationError),
        max_tries=5,
    )
    async def _call_bedrock(
        self, layers: list[LayerInformationCounterfeits], max_tokens: int = 4096, temperature: float = 0.0
    ) -> list[LayerInformationCounterfeits]:
        """Send one batch of items to Bedrock and parse the counterfeit rewrites, retrying on transient errors.

        Args:
            layers (list[LayerInformationCounterfeits]): Batch of items to rewrite.
            max_tokens (int): Max tokens for the Bedrock response. Defaults to 4096.
            temperature (float): Sampling temperature. Defaults to 0.0.

        Returns:
            list[LayerInformationCounterfeits]: The same items with `counterfeit_text` filled in.

        Raises:
            ValueError: If the response was truncated or the number of predictions doesn't match the input.
        """
        message = await self.bedrock_client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
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
                    "content": json.dumps([asdict(t) for t in layers]),
                }
            ],
        )
        if message.stop_reason == "max_tokens":
            raise ValueError(
                f"Bedrock response truncated (max_tokens={max_tokens}): increase max_tokens or reduce batch size"
            )

        tool_result = next(b for b in message.content if b.type == "tool_use")
        predictions = [
            LayerInformationCounterfeits(**item) for item in (tool_result.input.get("predictions", None) or [])
        ]

        if len(predictions) != len(layers):
            raise ValueError(f"Wrong number of predictions {len(layers)=}, {len(predictions)=}")

        # The tool schema only asks Bedrock for the text/class fields; carry the routing info
        # (filename/borehole_index/layer_index) over from the input instead of round-tripping it.
        for prediction, layer in zip(predictions, layers, strict=True):
            prediction.filename = layer.filename
            prediction.borehole_index = layer.borehole_index
            prediction.layer_index = layer.layer_index

        return predictions

    async def _process_batch(self, layers: list[LayerInformationCounterfeits]) -> list[LayerInformationCounterfeits]:
        """Process one batch through Bedrock, bounded by the concurrency semaphore.

        On failure, returns the batch unchanged (with `counterfeit_text` left as `None`) instead of raising.

        Args:
            layers (list[LayerInformationCounterfeits]): Batch of items to rewrite.

        Returns:
            list[LayerInformationCounterfeits]: The batch, rewritten where the call succeeded.
        """
        async with asyncio.Semaphore(self.max_concurrent_calls):
            try:
                return await self._call_bedrock(layers)
            except Exception as e:
                logger.warning(f"API call failed {e}")
                return layers

    async def process(
        self, layers: list[LayerInformationCounterfeits], batch_size: int = 5
    ) -> list[LayerInformationCounterfeits]:
        """Generate counterfeits for all items, split into concurrently-processed batches.

        Args:
            layers (list[LayerInformationCounterfeits]): Items to rewrite.
            batch_size (int): Number of items sent per Bedrock call. Defaults to 5.

        Returns:
            list[LayerInformationCounterfeits]: All items, in the same order, with counterfeits filled in.
        """
        batches = await tqdm_asyncio.gather(
            *[self._process_batch(layers=layers[i : i + batch_size]) for i in range(0, len(layers), batch_size)],
            desc="Classifying files",
        )
        return [item for batch in batches for item in batch]


def generate(
    samples: list[LayerInformation],
    classification_system_cls: type[ClassificationSystem],
    aws_model: AWSBedrockCounterfeits,
    seed: int = 0,
) -> list[LayerInformationCounterfeits]:
    """Generate a counterfeit rewrite for each sample, targeting a random other class.

    Args:
        samples (list[LayerInformation]): Ground truth layers to generate counterfeits for.
        classification_system_cls (type[ClassificationSystem]): Classification system defining the class set.
        aws_model (AWSBedrockCounterfeits): Bedrock client used to generate the counterfeit rewrites.
        seed (int): Seed for the random target-class assignment. Defaults to 0.

    Returns:
        list[LayerInformationCounterfeits]: One counterfeit item per input sample.
    """
    counterfeit_classes = list(classification_system_cls.get_enum())
    rnd = np.random.RandomState(seed=seed)
    rnd_class_samples = rnd.randint(low=0, high=len(counterfeit_classes), size=len(samples))

    return asyncio.run(
        aws_model.process(
            [
                LayerInformationCounterfeits(
                    ground_truth_text=layer.material_description,
                    counterfeit_text=None,
                    ground_truth_class=layer.ground_truth_class[0].name,
                    counterfeit_class=counterfeit_classes[rnd_class].name,
                    filename=layer.filename,
                    borehole_index=layer.borehole_index,
                    layer_index=layer.layer_index,
                )
                for layer, rnd_class in zip(samples, rnd_class_samples, strict=True)
            ]
        )
    )


def to_ground_truth(
    samples: list[LayerInformationCounterfeits], classification_system_cls: type[ClassificationSystem]
) -> dict[str, list[GroundTruthBorehole]]:
    """Rebuild counterfeit samples into the same dict[filename] -> list[GroundTruthBorehole] shape as GroundTruth.

    The counterfeit class is written to the single nested field this classification system reads from
    (its first `get_layer_ground_truth_keys()` group): `consolidated`/`unconsolidated` on the layer, or
    `metadata` on the borehole for document-level systems (e.g. `borehole_type`). Depth intervals and
    any other metadata are left at their defaults, since counterfeits carry no such information.

    Args:
        samples (list[LayerInformationCounterfeits]): Counterfeit samples to convert.
        classification_system_cls (type[ClassificationSystem]): Classification system that produced them.

    Returns:
        dict[str, list[GroundTruthBorehole]]: Same structure as `GroundTruth.ground_truth`.
    """
    root, field = classification_system_cls.get_layer_ground_truth_keys()[0]

    samples_by_file_and_borehole: dict[str, dict[int, list[LayerInformationCounterfeits]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        samples_by_file_and_borehole[sample.filename][sample.borehole_index].append(sample)

    return {
        filename: [
            GroundTruthBorehole(
                borehole_index=borehole_index,
                layers=[
                    GroundTruthLayer(
                        depth_interval=GroundTruthLayerDepth(),
                        material_description=sample.counterfeit_text,
                        consolidated=GroundTruthConsolidated(**{field: sample.counterfeit_class})
                        if root == "consolidated"
                        else None,
                        unconsolidated=GroundTruthUnconsolidated(**{field: sample.counterfeit_class})
                        if root == "unconsolidated"
                        else None,
                    )
                    for sample in borehole_samples
                ],
                metadata=GroundTruthMetadata(**{field: borehole_samples[0].counterfeit_class})
                if root == "metadata"
                else GroundTruthMetadata(),
            )
            for borehole_index, borehole_samples in boreholes.items()
        ]
        for filename, boreholes in samples_by_file_and_borehole.items()
    }


def main(
    ground_truth_path: Path,
    classification_system: str,
    examples_path: Path,
    output_folder: Path,
    n_samples: int = 10,
    seed: int = 0,
) -> None:
    """Load ground truth samples for a classification system and generate counterfeits for the train split.

    Args:
        ground_truth_path (Path): Path to the ground truth JSON file.
        classification_system (str): Name of the classification system to generate counterfeits for.
        examples_path (Path): Path to the YAML file of classification examples shown to Bedrock.
        output_folder (Path): Output JSON file path for the generated counterfeit ground truth.
        n_samples (int): Number of train samples to generate counterfeits for. Defaults to 10.
        seed (int): Seed for the random target-class assignment. Defaults to 0.
    """
    aws_model = AWSBedrockCounterfeits(examples_path=examples_path)

    # Step 1: Load ground truth examples for classes
    ground_truth = GroundTruth(ground_truth_path)
    classification_system_cls = ExistingClassificationSystems.get_classification_system_type(classification_system)
    gt_boreholes = GroundTruthBoreholeWithLanguage.from_ground_truth(ground_truth=ground_truth.ground_truth)
    samples = classification_system_cls.process(ground_truth=gt_boreholes)

    # Step 2: Generate counterfeit exmaples for classes
    counterfeit_samples = generate(
        samples=samples[:n_samples],
        classification_system_cls=classification_system_cls,
        aws_model=aws_model,
        seed=seed,
    )

    # Step 3: Save parsed outputs
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{classification_system}_counterfeit_ground_truth.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                filename: [borehole.model_dump() for borehole in boreholes]
                for filename, boreholes in to_ground_truth(counterfeit_samples, classification_system_cls).items()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Wrote %d counterfeit samples to %s", len(counterfeit_samples), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=Path("data/thurgau_ground_truth.json"),
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=Path("bedrock/examples/bedrock_color_examples.yml"),
        help="Path to the YAML file of classification examples shown to Bedrock.",
    )
    parser.add_argument(
        "-cs",
        "--classification-system",
        choices=[system.name for system in ExistingClassificationSystems],
        default="color",
        help="The classification system to generate counterfeits for.",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=10,
        help="Number of train samples to generate counterfeits for.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=Path("data/bert_extra"),
        help="Output folder for the generated counterfeit ground truth.",
    )

    args = parser.parse_args()

    main(args.ground_truth_path, args.classification_system, args.examples_path, args.output_folder, args.n_samples)
