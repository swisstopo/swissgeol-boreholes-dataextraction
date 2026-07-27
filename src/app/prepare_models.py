"""Download full models from HuggingFace and split them into backbone + head format.

It downloads each full fine-tuned model, extracts the frozen backbone (shared across all systems) and the task-specific
head weights, and writes them to the directory layout expected by classify_all.py:


    models/backbone/backbone.safetensors  — shared frozen BERT encoder weights
    models/backbone/                      — tokenizer files
    models/model_head_1/model.safetensors
    models/model_head_1/config.json
    ...
    models/model_head_n/model.safetensors
    models/model_head_n/config.json
The backbone is extracted from the first model and assumed to be identical across all
fine-tuned models (they all share the same frozen pre-trained encoder).
"""

import logging
from pathlib import Path

import backoff
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# HuggingFace model IDs mapped to their output head directory.
# The backbone is extracted from the first entry and reused for all subsequent ones.
MODELS: list[tuple[str, Path]] = [
    ("swissgeol/accessory_components", Path("models/accessory_components_head")),
    ("swissgeol/alteration_degree_consolidated", Path("models/alteration_degree_consolidated_head")),
    ("swissgeol/cementation", Path("models/cementation_head")),
    ("swissgeol/color", Path("models/color_head")),
    ("swissgeol/debris", Path("models/debris_head")),
    ("swissgeol/en_main", Path("models/en_main_head")),
    ("swissgeol/borehole_type", Path("models/borehole_type")),
    ("swissgeol/grain_angularity", Path("models/grain_angularity_head")),
    ("swissgeol/grain_shape", Path("models/grain_shape_head")),
    ("swissgeol/lithology", Path("models/lithology_head")),
    ("swissgeol/mineral_components", Path("models/mineral_components_head")),
    ("swissgeol/organic_components", Path("models/organic_components_head")),
    ("swissgeol/uscs", Path("models/uscs_head")),
]

BACKBONE_DIR = Path("models/backbone")

# Parameter prefixes belonging to the task-specific head.
# Mirrors unfreeze_layers: ["classifier", "pooler", "layer_11"] in the BERT config YAMLs.
_HEAD_PREFIXES = ("classifier.", "bert.pooler.", "bert.encoder.layer.11.")


def _is_head_param(name: str) -> bool:
    """Check if a parameter name belongs to the task-specific head."""
    return any(name.startswith(prefix) for prefix in _HEAD_PREFIXES)


@backoff.on_exception(backoff.expo, OSError, max_tries=5, factor=30)
def _load_model(hf_model_id: str) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(hf_model_id)


@backoff.on_exception(backoff.expo, OSError, max_tries=5, factor=30)
def _load_tokenizer(hf_model_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(hf_model_id)


def prepare_models() -> None:
    """Download and split all models listed in MODELS."""
    backbone_saved = False

    for hf_model_id, head_dir in MODELS:
        logger.info(f"Downloading {hf_model_id} tokenizer ...")
        tokenizer = _load_tokenizer(hf_model_id)
        logger.info(f"Downloading {hf_model_id} model ...")
        model = _load_model(hf_model_id)

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
