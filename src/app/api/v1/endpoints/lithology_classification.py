"""Endpoint for classifying lithological attributes from plain text descriptions using BERT."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

from app.common.aws import download_backbone_from_s3, download_model_from_s3
from app.common.schemas import ClassifyLithologyRequest, ClassifyLithologyResponse

if TYPE_CHECKING:
    from classification.models.model import BertModel

logger = logging.getLogger(__name__)

# Environment variable names for local full-model paths (BERT_LOCAL=true, BERT_SPLIT=false).
_MODEL_PATH_ENV_VARS = {
    "lithology": "BERT_MODEL_PATH_LITHOLOGY",
    "en_main": "BERT_MODEL_PATH_EN_MAIN",
}

# Environment variable names for local split-model head paths (BERT_LOCAL=true, BERT_SPLIT=true).
_MODEL_PATH_HEAD_ENV_VARS = {
    "lithology": "BERT_MODEL_PATH_LITHOLOGY_HEAD",
    "en_main": "BERT_MODEL_PATH_EN_MAIN_HEAD",
}

# Environment variable names for S3 full-model key prefixes (BERT_LOCAL=false, BERT_SPLIT=false).
_MODEL_S3_KEY_ENV_VARS = {
    "lithology": "BERT_MODEL_S3_KEY_LITHOLOGY",
    "en_main": "BERT_MODEL_S3_KEY_EN_MAIN",
}

# Environment variable names for S3 split-model head key prefixes (BERT_LOCAL=false, BERT_SPLIT=true).
_MODEL_S3_KEY_HEAD_ENV_VARS = {
    "lithology": "BERT_MODEL_S3_KEY_LITHOLOGY_HEAD",
    "en_main": "BERT_MODEL_S3_KEY_EN_MAIN_HEAD",
}

# Where downloaded models are cached on disk between warm Lambda invocations.
_LOCAL_MODEL_CACHE_DIR = Path("/tmp/bert_models")


def _resolve_model_path(classification_system_name: str) -> str:
    """Resolve the local path to the fine-tuned BERT model.

    Behaviour is controlled by two env vars:
      BERT_LOCAL — true: load from a local directory | false: download from S3 (default)
      BERT_SPLIT — true: use backbone + task head | false: use a single full model (default)

    Args:
        classification_system_name (str): One of 'lithology', 'en_main'.

    Returns:
        str: Absolute path to a local directory containing the model files.

    Raises:
        HTTPException: If the required env var is not set or the path does not exist.
    """
    bert_local = os.environ.get("BERT_LOCAL", "false").lower() == "true"
    bert_split = os.environ.get("BERT_SPLIT", "false").lower() == "true"

    if bert_local:
        env_var = (
            _MODEL_PATH_HEAD_ENV_VARS[classification_system_name]
            if bert_split
            else _MODEL_PATH_ENV_VARS[classification_system_name]
        )
        local_path = os.environ.get(env_var)
        if local_path and Path(local_path).exists():
            logger.info(f"Loading {classification_system_name} model from {local_path}")
            return local_path
        raise HTTPException(
            status_code=500,
            detail=f"No model for '{classification_system_name}': {env_var} is not set or path does not exist.",
        )

    env_var = (
        _MODEL_S3_KEY_HEAD_ENV_VARS[classification_system_name]
        if bert_split
        else _MODEL_S3_KEY_ENV_VARS[classification_system_name]
    )
    s3_key = os.environ.get(env_var)
    if s3_key:
        local_dir = _LOCAL_MODEL_CACHE_DIR / classification_system_name
        logger.info(f"Downloading {classification_system_name} model from S3 key: {s3_key}")
        download_model_from_s3(s3_key, local_dir)
        return str(local_dir)
    raise HTTPException(
        status_code=500,
        detail=f"No model for '{classification_system_name}': {env_var} is not set.",
    )


def _resolve_backbone_path() -> str | None:
    """Return the local path to the shared backbone weights, or None if BERT_SPLIT is not enabled.

    When BERT_SPLIT=true:
      BERT_LOCAL=true  — reads BERT_MODEL_PATH_BACKBONE
      BERT_LOCAL=false — downloads from BERT_MODEL_S3_KEY_BACKBONE on first use
    """
    if os.environ.get("BERT_SPLIT", "false").lower() != "true":
        return None

    if os.environ.get("BERT_LOCAL", "false").lower() == "true":
        local_path = os.environ.get("BERT_MODEL_PATH_BACKBONE")
        if local_path and Path(local_path).exists():
            return local_path
        return None

    s3_key = os.environ.get("BERT_MODEL_S3_KEY_BACKBONE")
    if s3_key:
        local_path = _LOCAL_MODEL_CACHE_DIR / "backbone" / "backbone.safetensors"
        download_backbone_from_s3(s3_key, local_path)
        return str(local_path)
    return None


def load_models() -> dict[str, BertModel]:
    """Load all BERT models. Called once at application startup via the lifespan.

    Returns:
        dict mapping classification system name to its loaded BertModel.

    Raises:
        HTTPException: If a required env var is missing or a path does not exist.
    """
    from classification.models.model import BertModel
    from classification.utils.classification_classes import ExistingClassificationSystems

    backbone_path = _resolve_backbone_path()
    models = {}
    for system_name in _MODEL_PATH_ENV_VARS:
        model_path = _resolve_model_path(system_name)
        classification_system = ExistingClassificationSystems.get_classification_system_type(system_name)
        models[system_name] = BertModel(model_path, classification_system, backbone_path=backbone_path)
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
