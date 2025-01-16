"""Contains utility functions for depth column entries."""


def value_as_float(string_value: str) -> float:  # noqa: D103
    """Converts a string to a float."""
    # OCR sometimes tends to miss the decimal comma
    return abs(float(string_value))
