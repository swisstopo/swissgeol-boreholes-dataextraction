"""This module contains functionalities for language detection of a document."""

import pymupdf
from fast_langdetect import LangDetectConfig, LangDetector


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


def detect_language_of_text(
    text: str,
    default_language: str,
    supported_languages: list,
    context_window: int = 80,
    n_windows: int = 5,
) -> str:
    """Detects the language of a text.

    Args:
        text (str): The text to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list): A list of supported languages.
        context_window (int): Size of context window for text detection.
        n_windows (int): Number of context windows for language detection. Defaults to 5.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    # Define set of segements for context window
    config = LangDetectConfig(max_input_length=context_window)
    detector = LangDetector(config)
    bins_size = len(text) // n_windows

    # Perform language detection on windows and extract top-1 languages
    languages = [
        detector.detect(
            text=text[i * bins_size : min(i * bins_size + context_window, len(text))],
            k=1,  # Return only top 1 lang
            model="lite",  # Lite model to speed up
        )[0].get("lang", "")
        for i in range(n_windows)
    ]

    # Perofrm majority voting across windows
    language = max(set(languages), key=languages.count)

    # Return language if part of supported otherwise default
    return language if language in supported_languages else default_language
