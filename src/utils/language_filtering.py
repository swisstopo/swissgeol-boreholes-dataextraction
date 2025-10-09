"""This module contains functionalities for text processing and normalzation."""

import re


def normalize_spaces(text: str) -> str:
    """Normalize whitespace and remove leading/trailing spaces.

    Args:
        text (str): Text to be normlized.

    Returns:
        str: Normalized text (single spaces, no leading/trailing spaces).
    """
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.strip()
    return cleaned


def match_any_keyword(text: str, keywords: list[str], start: bool = False, end: bool = False) -> re.Match | None:
    """Search for the first occurrence of any keyword from a predefined list in a text.

    The search is case-insensitive and treats keywords as literals. You can control whether
    the keyword must appear at the `start` or `end` of a word.

    Args:
        text (str): The text to search within.
        keywords (list[str]): A list of keywords to look for.
        start (bool, optional): If True, the word must start with the keyword. Defaults to False.
        end (bool, optional): If True, the word must end with the keyword. Defaults to False.

    Returns:
        re.Match | None: The first match object found in the text, or None if no keyword is present.
    """
    # Build a regex pattern that matches keywords
    reg_start = "" if start else r"\w*"
    reg_end = "" if end else r"\w*"
    pattern = r"\b" + reg_start + "(?:" + "|".join(re.escape(kw) for kw in keywords) + r")" + reg_end + r"\b"
    return re.search(pattern, text, re.IGNORECASE)


def remove_any_keyword(text: str, keywords: list[str]) -> str:
    """Remove all occurrences of specified keywords from the text.

    The removal is case-insensitive and literal — special characters in keywords
    (such as °, ., or º) are safely escaped. Each keyword is removed if it appears
    as a standalone term or directly attached to punctuation, without requiring
    strict word boundaries. This allows matching cases like “N°”, “Nr”, “No.”, etc.

    Args:
        text (str): The input text to clean.
        keywords (list[str]): List of keywords to remove. Each keyword is treated
            as a literal string.

    Returns:
        str: The cleaned text with all matching keywords removed.
    """
    # Build regex pattern for keywords (escaped and followed by a word boundary)
    pattern = "(" + "|".join(r"(?<!\w)" + re.escape(kw) + r"(?=\W|\d|$)" for kw in keywords) + ")"
    # Remove matched keywords (case-insensitive)
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return cleaned
