"""This module contains functionalities for text processing and normalization."""

import re


def normalize_spaces(text: str) -> str:
    """Normalize whitespace and remove leading/trailing spaces.

    Args:
        text (str): Text to be normalized.

    Returns:
        str: Normalized text (single spaces, no leading/trailing spaces).
    """
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.strip()
    return cleaned


def remove_any_keyword(text: str, keywords: list[str]) -> str:
    """Remove all occurrences of specified keywords as literal strings from the text.

    Args:
        text (str): The input text to clean.
        keywords (list[str]): List of keywords to remove.

    Returns:
        str: The cleaned text with all matching keywords removed.
    """
    # Build regex pattern for keywords
    pattern = "(" + "|".join(r"(?<!\w)" + re.escape(kw) + r"(?=\W|\d|$)" for kw in keywords) + ")"
    # Remove matched keywords (case-insensitive)
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return cleaned
