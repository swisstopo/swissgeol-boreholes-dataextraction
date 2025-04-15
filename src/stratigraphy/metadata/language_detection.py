"""This module contains functionalities for language detection of a document."""

import fitz
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def extract_text_from_document(doc: fitz.Document) -> str:
    """Extracts and processes text from a document.

    Args:
        doc (fitz.Document): The document to extract text from.

    Returns:
        str: The extracted and processed text.
    """
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.replace("\n", " ")

    # remove all numbers and special characters from text
    return "".join(e for e in text if (e.isalnum() or e.isspace()) and not e.isdigit())


def detect_language_of_document(doc: fitz.Document, default_language: str, supported_languages: list) -> str:
    """Detects the language of a document.

    Args:
        doc (fitz.Document): The document to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list): A list of supported languages.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    text = extract_text_from_document(doc)
    return detect_language_of_text(text, default_language, supported_languages)


def detect_language_of_text(text: str, default_language: str, supported_languages: list) -> str:
    """Detects the language of a text.

    Args:
        text (str): The text to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list): A list of supported languages.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    try:
        language = detect(text)
    except LangDetectException:
        language = default_language

    if language not in supported_languages:
        language = default_language
    return language
