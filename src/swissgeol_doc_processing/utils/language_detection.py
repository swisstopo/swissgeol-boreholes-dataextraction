"""This module contains functionalities for language detection of a document."""

import pymupdf
from fast_langdetect import LangDetectConfig, LangDetector

config = LangDetectConfig(max_input_length=1000)
detector = LangDetector(config)


def extract_text_from_document(doc: pymupdf.Document) -> str:
    """Extracts and processes text from a document.

    Args:
        doc (pymupdf.Document): The document to extract text from.

    Returns:
        str: The extracted and processed text.
    """
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.replace("\n", " ")

    # remove all numbers and special characters from text
    return "".join(e for e in text if (e.isalnum() or e.isspace()) and not e.isdigit())


def detect_language_of_document(doc: pymupdf.Document, default_language: str, supported_languages: list) -> str:
    """Detects the language of a document.

    Args:
        doc (pymupdf.Document): The document to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list): A list of supported languages.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    text = extract_text_from_document(doc)
    return detect_language_of_text(text, default_language, supported_languages)


def detect_language_of_text(text: str, default_language: str, supported_languages: list, seed: int = 0) -> str:
    """Detects the language of a text.

    Args:
        text (str): The text to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list): A list of supported languages.
        seed (int): Define seed for language detection.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    # Detect language using lite detection model
    languages = detector.detect(text, model="lite")

    # No language detected, use default
    if len(languages) == 0:
        return default_language

    # Ensure order and take highest confidence lang
    languages.sort(key=lambda x: x["score"], reverse=True)
    language = languages[0].get("lang", None)

    return language if language in supported_languages else default_language
