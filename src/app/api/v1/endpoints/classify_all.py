"""Endpoint for classifying all lithological attributes in a single backbone forward pass."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.common.schemas import ClassifyRequest, ClassifyResponse
from classification.utils.datasets.classification import ClassificationTask

if TYPE_CHECKING:
    from classification.models.model import BertModel

logger = logging.getLogger(__name__)

_BACKBONE_PATH = Path("models/backbone/backbone.safetensors")
_TOKENIZER_PATH = Path("models/backbone")
_MODEL_PATHS = {
    "accessory_components": Path("models/accessory_components_head"),
    "alteration_degree_consolidated": Path("models/alteration_degree_consolidated_head"),
    "cementation": Path("models/cementation_head"),
    "color": Path("models/color_head"),
    "debris": Path("models/debris_head"),
    "en_main": Path("models/en_main_head"),
    "en_secondary": Path("models/en_secondary_head"),
    "grain_angularity": Path("models/grain_angularity_head"),
    "grain_shape": Path("models/grain_shape_head"),
    "lithology": Path("models/lithology_head"),
    "mineral_components": Path("models/mineral_components_head"),
    "organic_components": Path("models/organic_components_head"),
    "uscs": Path("models/uscs_head"),
}

# Tasks returned for consolidated rock (lithology head does NOT predict "unconsolidated").
_CONSOLIDATED_TASKS: frozenset[str] = frozenset(
    {
        "lithology",
        "alteration_degree_consolidated",
        "cementation",
        "color",
        "mineral_components",
        "accessory_components",
    }
)

# Tasks returned for unconsolidated sediment (lithology head predicts "unconsolidated").
_UNCONSOLIDATED_TASKS: frozenset[str] = frozenset(
    {
        "en_main",
        "en_secondary",
        "uscs",
        "debris",
        "color",
        "grain_angularity",
        "grain_shape",
        "organic_components",
    }
)


def load_models() -> dict[str, BertModel]:
    """Load all BERT models once at application startup via the lifespan.

    Returns:
        dict mapping task name to its loaded BertModel.
    """
    from classification.models.model import BertModel
    from classification.utils.datasets import ExistingClassificationSystems

    models = {}
    for system_name, model_path in _MODEL_PATHS.items():
        classification_system = ExistingClassificationSystems.get_classification_system_type(system_name)
        models[system_name] = BertModel(
            model_path, classification_system, backbone_path=_BACKBONE_PATH, tokenizer_path=_TOKENIZER_PATH
        )
        logger.info(f"Loaded {system_name} model from {model_path}")
    return models


def classify(request: ClassifyRequest, bert_models: dict[str, BertModel]) -> ClassifyResponse:
    """Classify a description across all relevant tasks via a two-step backbone pass.

    Step 1: run the lithology model to obtain the shared backbone embedding and determine
    whether the material is consolidated or unconsolidated.
    Step 2: pass the saved embedding directly to each relevant task head — no second
    backbone forward pass.

    Args:
        request: Classification request containing a plain-text material description.
        bert_models: Models loaded at startup via the lifespan, keyed by task name.

    Returns:
        ClassifyResponse with the predicted class (or classes, for multi-label/rank tasks) for every task
        relevant to the inferred rock type; irrelevant tasks are left as `None`.
    """
    from classification.utils.datasets.lithology import LithologySystem

    # Step 1: shared backbone (layers 0–10) — runs once regardless of how many task heads follow.
    lithology_model = bert_models["lithology"]
    shared_hidden_states, extended_mask = lithology_model.compute_shared_embedding(request.description)

    # Lithology head (layer 11 + pooler + classifier) determines consolidated vs unconsolidated.
    lithology_class = lithology_model.predict_from_embedding(shared_hidden_states, extended_mask)[0]
    is_unconsolidated = lithology_class == LithologySystem.LithologyClasses.unconsolidated

    relevant_tasks = _UNCONSOLIDATED_TASKS if is_unconsolidated else _CONSOLIDATED_TASKS

    # Step 2: fan out to each relevant task head using the same shared backbone output.
    predictions = dict.fromkeys(relevant_tasks)
    for task_name in relevant_tasks:
        if task_name not in bert_models:
            logger.warning(f"Task '{task_name}' not loaded, skipping.")
            continue
        if task_name == "lithology":
            predictions["lithology"] = lithology_class
            continue
        model = bert_models[task_name]
        classes = model.predict_from_embedding(shared_hidden_states, extended_mask)
        predictions[task_name] = (
            classes
            if model.classification_system.classification_task() != ClassificationTask.single_label
            else classes[0]
        )

    return ClassifyResponse(**predictions)
