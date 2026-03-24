"""This module contains functionalities for text processing and normalization."""

import re


def remove_scale(text: str) -> str:
    """Remove scale notations from a text string.

    The function identifies and removes map or drawing scale expressions that
    start with an optional 'M' or 'M.', possibly followed by spaces, and ending
    with '1:' and one to three digits (e.g., 'M. 1:100', 'M 1:50', '1:20').

    Args:
        text (str): Input text potentially containing scale expressions.

    Returns:
        str: The cleaned text with scale notations replaced by a single space.
    """
    return re.sub(r"(?:M\.?\s*)?1:\d{1,3}", "", text)


def remove_in_parenthesis(text: str) -> str:
    """Remove all text enclosed in parentheses, including the parentheses themselves.

    Args:
        text (str): Input text containing parenthetical expressions.

    Returns:
        str: The text with all parenthetical content removed and replaced by a space.
    """
    return re.sub(r"\(.*?\)", "", text)


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


def match_any_keyword(
    text: str,
    keywords: list[str],
    start: bool = False,
    end: bool = False,
    ignore_case: bool = True,
) -> re.Match | None:
    """Search for the first occurrence of any keyword from a predefined list in a text.

    Keywords are treated as **raw regex patterns**. Callers are responsible for escaping
    metacharacters.

    Args:
        text (str): The text to search within.
        keywords (list[str]): A list of regex patterns to look for. Metacharacters must be
            escaped by the caller.
        start (bool, optional): If True, the word must start with the keyword. Defaults to False.
        end (bool, optional): If True, the word must end with the keyword. Defaults to False.
        ignore_case (bool, optional): If True, keyword matching is case insensitive. Defaults to True.

    Returns:
        re.Match | None: The first match object found in the text, or None if no keyword is present.
    """
    # Build a regex pattern that matches keywords
    if keywords is None or len(keywords) == 0:
        return None
    reg_start = "" if start else r"\w*"
    reg_end = "" if end else r"\w*"
    pattern = r"\b" + reg_start + "(?:" + "|".join(keywords) + r")" + reg_end + r"\b"

    return re.search(pattern, text, flags=re.IGNORECASE if ignore_case else re.NOFLAG)


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
