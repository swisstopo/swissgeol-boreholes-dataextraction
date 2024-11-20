"""Contains utility functions for depth column entries."""

import re


def value_as_float(string_value: str) -> float:  # noqa: D103
    """Converts a string to a float."""
    # OCR sometimes tends to miss the decimal comma
    parsed_text = re.sub(r"^-?([0-9]+)([0-9]{2})", r"\1.\2", string_value)
    return abs(float(parsed_text))
