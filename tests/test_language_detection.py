"""This module contains functionalities for testing language detection."""

import pymupdf
import pytest

from swissgeol_doc_processing.utils.language_detection import detect_language_of_document, detect_language_of_text

ENGLISH_GERMAN_TEXT = (
    "Dies ist ein Test zur Überprüfung der Textsprache. This is a test to check text language. "
    "However, most of this text is written in english and not in german"
)
GERMAN_TEXT = "Dies ist ein einfaches deutsches Dokument."
ENGLISH_TEXT = "This is a simple English document."
ITALIAN_TEXT = "Questo è un semplice documento italiano."


@pytest.mark.parametrize(
    "text, default_language, supported_languages, context_window, n_windows, expected",
    [
        pytest.param(GERMAN_TEXT, "de", ["de", "en"], 5, 1, "de", id="text-de"),
        pytest.param(ENGLISH_TEXT, "de", ["de", "en"], 5, 1, "en", id="text-en"),
        pytest.param(ITALIAN_TEXT, "de", ["de", "en"], 5, 1, "de", id="text-it-default-de"),
        pytest.param(ITALIAN_TEXT, "de", ["de", "en", "it"], 5, 1, "it", id="text-it"),
        pytest.param(ENGLISH_GERMAN_TEXT, "de", ["de", "en"], 5, 1, "de", id="text-multi-de-single"),
        pytest.param(ENGLISH_GERMAN_TEXT, "de", ["de", "en"], 5, 5, "en", id="text-multi-en-multi"),
    ],
)
def test_predict_language(
    text: str,
    default_language: str,
    supported_languages: list[str],
    context_window: int,
    n_windows: int,
    expected: str,
) -> None:
    """Test language prediction from text.

    Args:
        text (str): Text to use for detection.
        default_language (str): Fallback language when detection yields an unsupported language.
        supported_languages (list[str]): List of accepted language codes.
        context_window (int): Size of the context window.
        n_windows (int): Number of windows for language detection.
        expected (str): Expected language.
    """
    assert (
        detect_language_of_text(
            text,
            default_language=default_language,
            supported_languages=supported_languages,
            n_windows=n_windows,
            context_window=context_window,
        )
        == expected
    )


@pytest.mark.parametrize(
    "text, default_language, supported_languages, expected",
    [
        pytest.param(GERMAN_TEXT, "de", ["de", "en"], "de", id="text-de"),
        pytest.param(ENGLISH_TEXT, "de", ["de", "en"], "en", id="text-en"),
        pytest.param(ITALIAN_TEXT, "de", ["de", "en"], "de", id="text-it-default-de"),
        pytest.param(ITALIAN_TEXT, "de", ["de", "en", "it"], "it", id="text-it"),
        pytest.param(ENGLISH_GERMAN_TEXT, "de", ["de", "en", "it"], "en", id="text-multi-en"),
    ],
)
def test_detect_language_of_document(
    text: str, default_language: str, supported_languages: list[str], expected: str
) -> None:
    """Test language detection on a PDF document.

    Args:
        text (str): The text content to insert into the test PDF document.
        default_language (str): Fallback language when detection yields an unsupported language.
        supported_languages (list[str]): List of accepted language codes.
        expected (str): Expected detected language code.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)

    assert detect_language_of_document(doc, default_language, supported_languages) == expected
