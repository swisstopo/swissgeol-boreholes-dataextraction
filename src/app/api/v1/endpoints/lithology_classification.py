"""Endpoint for classifying lithological attributes from plain text descriptions using BERT."""

import logging
import os

from app.common.schemas import ClassifyLithologyRequest, ClassifyLithologyResponse
from classification.models.model import BertModel
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.file_utils import read_params

logger = logging.getLogger(__name__)

_BERT_CONFIG_PATHS: dict = read_params("classifier_config_paths.yml")["bert"]

# Environment variable names for model paths, one per classification system.
_MODEL_PATH_ENV_VARS = {
    "uscs": "BERT_MODEL_PATH_USCS",
    "lithology": "BERT_MODEL_PATH_LITHOLOGY",
    "en_main": "BERT_MODEL_PATH_EN_MAIN",
}

# Module-level model cache keyed by (classification_system_name, resolved_model_path).
# Avoids reloading the BERT model on every request.
_model_cache: dict[tuple[str, str], BertModel] = {}


def _resolve_model_path(classification_system_name: str) -> str:
    """Resolve the model path from environment variable or fall back to the config default.

    Args:
        classification_system_name (str): One of 'uscs', 'lithology', 'en_main'.

    Returns:
        str: The resolved model path.
    """
    env_var = _MODEL_PATH_ENV_VARS[classification_system_name]
    model_path = os.environ.get(env_var)
    if model_path:
        return model_path
    classification_system = ExistingClassificationSystems.get_classification_system_type(classification_system_name)
    config_file = _BERT_CONFIG_PATHS[classification_system.get_name()]
    model_path = read_params(config_file)["model_path"]
    logger.warning(
        f"{env_var} is not set. Falling back to the base pre-trained model '{model_path}', "
        "which is not fine-tuned and will produce meaningless predictions."
    )
    return model_path


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
