"""language module."""

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def detect_language(text, default_language, supported_languages):
    """_summary_.

    Args:
        text (_type_): _description_
        default_language (_type_): _description_
        supported_languages (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        language = detect(text)
    except LangDetectException:
        language = default_language

    if language not in supported_languages:
        language = default_language
    return language
