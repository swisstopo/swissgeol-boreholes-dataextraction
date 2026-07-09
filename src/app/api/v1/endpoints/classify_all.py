"""Endpoint for classifying all lithological attributes in a single backbone forward pass."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.common.schemas import ClassifyRequest, ClassifyResponse

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
    }
)

# Tasks returned for unconsolidated sediment (lithology head predicts "unconsolidated").
_UNCONSOLIDATED_TASKS: frozenset[str] = frozenset(
    {
        "en_main",
        "uscs",
        "debris",
        "color",
        "grain_angularity",
        "grain_shape",
        "organic_components",
        "accessory_components",
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
    """Classify a description across all relevant tasks.

    The lithology head determines whether the material is consolidated or unconsolidated;
    only tasks relevant to that rock type are returned.

    Args:
        request: Classification request containing a plain-text material description.
        bert_models: Models loaded at startup via the lifespan, keyed by task name.

    Returns:
        ClassifyResponse with a mapping of task name to predicted class name (string) or
        class names (list of strings for multi-label tasks).
    """
    from classification.utils.datasets.lithology import LithologySystem

    lithology_model = bert_models["lithology"]
    lithology_class = lithology_model.predict_class(request.description)[0]
    is_unconsolidated = lithology_class == LithologySystem.LithologyClasses.unconsolidated

    relevant_tasks = _UNCONSOLIDATED_TASKS if is_unconsolidated else _CONSOLIDATED_TASKS

    predictions: dict[str, str | list[str]] = {}
    for task_name in relevant_tasks:
        if task_name not in bert_models:
            logger.warning(f"Task '{task_name}' not loaded, skipping.")
            continue
        model = bert_models[task_name]
        classes = model.predict_class(request.description)
        if model.classification_system.is_multi_label():
            predictions[task_name] = [c.name for c in classes]
        else:
            predictions[task_name] = classes[0].name

    return ClassifyResponse(predictions=predictions)
