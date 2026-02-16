"""This module contains functionalities for language detection of a document."""

import re

import pymupdf
from fast_langdetect import LangDetectConfig, LangDetector

# Define the language detection mpdule (build it once)
config = LangDetectConfig(max_input_length=None)
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


def detect_language_of_document(doc: pymupdf.Document, default_language: str, supported_languages: list[str]) -> str:
    """Detects the language of a document.

    Args:
        doc (pymupdf.Document): The document to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list[str]): A list of supported languages.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    text = extract_text_from_document(doc)
    return detect_language_of_text(text, default_language, supported_languages)


def detect_language_of_text(
    text: str,
    default_language: str,
    supported_languages: list[str],
    context_window: int = 10,
    n_windows: int = 5,
) -> str:
    """Detects the language of a text.

    The context window is based on the number of words to sample. If the number of context window is larger
    than the number of possible non overlapping interval, the number of windows is reduced.

    Args:
        text (str): The text to detect the language of.
        default_language (str): The default language to use if the language detection fails.
        supported_languages (list[str]): A list of supported languages.
        context_window (int): Size of context window for text detection. Default to 10 (80 characters on average).
        n_windows (int): Number of context windows for language detection. Defaults to 5.

    Returns:
        str: The detected language of the document. One of supported_languages.
    """
    # Normalize spaces and split words
    text_words = re.sub(" +", " ", text.strip()).split(" ")

    # Ensure at least a single window exists and do not overlap with others
    n_window_min = max(len(text_words) // context_window, 1)
    n_windows = min(n_windows, n_window_min)
    bins_size = len(text_words) // n_windows

    # Perform language detection on windows and extract top-1 languages.
    # detector.detect always returns a list of candidates ordered by score.
    languages = [
        detector.detect(
            # Merge words from the i-th window to form text
            " ".join(text_words[i * bins_size : min(i * bins_size + context_window, len(text_words))]),
            # Return only top 1 lang
            k=1,
            # Lite model to speed up
            model="lite",
        )[0].get("lang", "")
        for i in range(n_windows)
    ]

    # Perform majority voting across windows
    language = max(set(languages), key=languages.count)

    # Return language if part of supported otherwise default
    return language if language in supported_languages else default_language
