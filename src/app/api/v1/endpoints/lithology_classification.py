"""Endpoint for classifying lithological attributes from plain text descriptions using BERT."""

import logging
import os
from pathlib import Path

from fastapi import HTTPException

from app.common.aws import download_model_from_s3
from app.common.schemas import ClassifyLithologyRequest, ClassifyLithologyResponse
from classification.models.model import BertModel
from classification.utils.classification_classes import ExistingClassificationSystems

logger = logging.getLogger(__name__)

# Environment variable names for model paths (local directory, takes priority over S3).
_MODEL_PATH_ENV_VARS = {
    "uscs": "BERT_MODEL_PATH_USCS",
    "lithology": "BERT_MODEL_PATH_LITHOLOGY",
    "en_main": "BERT_MODEL_PATH_EN_MAIN",
}

# Environment variable names for S3 key prefixes within the configured bucket.
# Example values: "lithology_models/best_model_lithology"
_MODEL_S3_KEY_ENV_VARS = {
    "uscs": "BERT_MODEL_S3_KEY_USCS",
    "lithology": "BERT_MODEL_S3_KEY_LITHOLOGY",
    "en_main": "BERT_MODEL_S3_KEY_EN_MAIN",
}

# Where downloaded models are cached on disk between warm Lambda invocations.
_LOCAL_MODEL_CACHE_DIR = Path("/tmp/bert_models")

# Module-level model cache keyed by (classification_system_name, resolved_model_path).
# Avoids reloading the BERT model on every request.
_model_cache: dict[tuple[str, str], BertModel] = {}


def _resolve_model_path(classification_system_name: str) -> str:
    """Resolve the local path to the fine-tuned BERT model.

    Resolution order:
    1. BERT_MODEL_PATH_<SYSTEM> env var — explicit local path (useful for local dev).
    2. BERT_MODEL_S3_KEY_<SYSTEM> env var — S3 key prefix; files are downloaded to
       /tmp/bert_models/<system>/ on first use and reused on subsequent calls.
    3. Raise a clear error — no silent fallback to the untrained base model.

    Args:
        classification_system_name (str): One of 'uscs', 'lithology', 'en_main'.

    Returns:
        str: Absolute path to a local directory containing the model files.

    Raises:
        HTTPException: If neither a local path nor an S3 key is configured.
    """
    local_path = os.environ.get(_MODEL_PATH_ENV_VARS[classification_system_name])
    if local_path:
        logger.info(f"Loading {classification_system_name} model from local path: {local_path}")
        return local_path

    s3_key = os.environ.get(_MODEL_S3_KEY_ENV_VARS[classification_system_name])
    if s3_key:
        local_dir = _LOCAL_MODEL_CACHE_DIR / classification_system_name
        logger.info(f"Downloading {classification_system_name} model from S3 key: {s3_key}")
        download_model_from_s3(s3_key, local_dir)
        return str(local_dir)

    local_env_var = _MODEL_PATH_ENV_VARS[classification_system_name]
    s3_env_var = _MODEL_S3_KEY_ENV_VARS[classification_system_name]
    raise HTTPException(
        status_code=500,
        detail=(
            f"No model configured for '{classification_system_name}'. "
            f"Set {local_env_var} (local directory path) or "
            f"{s3_env_var} (S3 key prefix, e.g. 'lithology_models/best_model_lithology')."
        ),
    )


def _get_bert_model(classification_system_name: str) -> BertModel:
    """Return a cached BertModel, loading it from disk on the first call.

    Args:
        classification_system_name (str): One of 'uscs', 'lithology', 'en_main'.

    Returns:
        BertModel: Ready-to-use model and tokenizer pair.
    """
    classification_system = ExistingClassificationSystems.get_classification_system_type(classification_system_name)
    model_path = _resolve_model_path(classification_system_name)
    cache_key = (classification_system_name, model_path)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = BertModel(model_path, classification_system)
    return _model_cache[cache_key]


def classify_lithology(request: ClassifyLithologyRequest) -> ClassifyLithologyResponse:
    """Classify a material description using the BERT model.

    Args:
        request (ClassifyLithologyRequest): The classification request containing a plain-text
            material description and the target classification system.

    Returns:
        ClassifyLithologyResponse: Predicted class name for the input description.
    """
    bert_model = _get_bert_model(request.classification_system)
    predicted_class = bert_model.predict_class(request.description)
    return ClassifyLithologyResponse(class_name=predicted_class.name)
