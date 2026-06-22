"""Download full models from HuggingFace and split them into backbone + head format.

It downloads each full fine-tuned model, extracts the frozen backbone (shared across all systems) and the task-specific
head weights, and writes them to the directory layout expected by lithology_classification.py:

    models/backbone/backbone.safetensors  — shared frozen BERT encoder weights
    models/backbone/                      — tokenizer files
    models/lithology_head/model.safetensors
    models/lithology_head/config.json
    models/en_main_head/model.safetensors
    models/en_main_head/config.json

The backbone is extracted from the first model and assumed to be identical across all
fine-tuned models (they all share the same frozen pre-trained encoder).
"""

import logging
from pathlib import Path

from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# HuggingFace model IDs mapped to their output head directory.
# The backbone is extracted from the first entry and reused for all subsequent ones.
MODELS: list[tuple[str, Path]] = [
    ("swissgeol/lithology", Path("models/lithology_head")),
    ("swissgeol/en_main", Path("models/en_main_head")),
]

BACKBONE_DIR = Path("models/backbone")

# Parameter prefixes belonging to the task-specific head.
# Mirrors unfreeze_layers: ["classifier", "pooler", "layer_11"] in the BERT config YAMLs.
_HEAD_PREFIXES = ("classifier.", "bert.pooler.", "bert.encoder.layer.11.")


def _is_head_param(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _HEAD_PREFIXES)


def prepare_models() -> None:
    """Download and split all models listed in MODELS."""
    backbone_saved = False

    for hf_model_id, head_dir in MODELS:
        logger.info(f"Downloading {hf_model_id} ...")
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        state_dict = model.state_dict()
        head_state = {k: v for k, v in state_dict.items() if _is_head_param(k)}
        backbone_state = {k: v for k, v in state_dict.items() if not _is_head_param(k)}

        if not backbone_saved:
            BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving backbone to {BACKBONE_DIR}/backbone.safetensors")
            save_file(backbone_state, BACKBONE_DIR / "backbone.safetensors")
            tokenizer.save_pretrained(BACKBONE_DIR)
            backbone_saved = True

        head_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving head to {head_dir}/")
        save_file(head_state, head_dir / "model.safetensors")
        model.config.save_pretrained(head_dir)

        logger.info(f"Done: {hf_model_id} → {head_dir}")


if __name__ == "__main__":
    prepare_models()
