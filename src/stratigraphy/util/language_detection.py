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


def detect_language_of_document(doc: fitz.Document) -> str:
    """Detects the language of a document.

    Args:
        doc (fitz.Document): The document to detect the language of.

    Returns:
        str: The detected language of the document. Either "de" or "fr".
    """
    text = extract_text_from_document(doc)
    try:
        language = detect(text)
    except LangDetectException:
        language = "de"  # TODO: default language should be read from config

    if language not in ["de", "fr"]:  # TODO: This should be read from the config
        language = "de"
    return language
