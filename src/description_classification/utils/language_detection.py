"""language module."""

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def detect_language(text: str, default_language: str, supported_languages: list[str]) -> str:
    """Detects the language of a given text.

    Args:
        text (str): The input text from which the language is detected.
        default_language (str): The language to return if detection fails or the detected language is unsupported.
        supported_languages (list[str]): A list of languages that are considered valid.

    Returns:
        str: The detected language if it is in the list of supported languages; otherwise, the default language.
    """
    try:
        language = detect(text)
    except LangDetectException:
        language = default_language

    if language not in supported_languages:
        language = default_language
    return language
