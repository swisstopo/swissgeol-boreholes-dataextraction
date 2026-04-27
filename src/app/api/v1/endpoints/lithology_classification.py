"""Endpoint for classifying lithological attributes from plain text descriptions using BERT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.common.schemas import ClassifyLithologyRequest, ClassifyLithologyResponse

if TYPE_CHECKING:
    from classification.models.model import BertModel

logger = logging.getLogger(__name__)

_BACKBONE_PATH = Path("models/backbone/backbone.safetensors")
_MODEL_PATHS = {
    "lithology": Path("models/lithology_head"),
    "en_main": Path("models/en_main_head"),
}


def load_models() -> dict[str, BertModel]:
    """Load all BERT models. Called once at application startup via the lifespan.

    Returns:
        dict mapping classification system name to its loaded BertModel.
    """
    from classification.models.model import BertModel
    from classification.utils.classification_classes import ExistingClassificationSystems

    models = {}
    for system_name, model_path in _MODEL_PATHS.items():
        classification_system = ExistingClassificationSystems.get_classification_system_type(system_name)
        models[system_name] = BertModel(model_path, classification_system, backbone_path=_BACKBONE_PATH)
        logger.info(f"Loaded {system_name} model from {model_path}")
    return models


def classify_lithology(
    request: ClassifyLithologyRequest, bert_models: dict[str, BertModel]
) -> ClassifyLithologyResponse:
    """Classify a material description using the BERT model.

    Args:
        request (ClassifyLithologyRequest): The classification request containing a plain-text
            material description and the target classification system.
        bert_models: Models loaded at startup via the lifespan, keyed by classification system name.

    Returns:
        ClassifyLithologyResponse: Predicted class name for the input description.
    """
    bert_model = bert_models[request.classification_system]
    predicted_class = bert_model.predict_class(request.description)
    return ClassifyLithologyResponse(class_name=predicted_class.name)
