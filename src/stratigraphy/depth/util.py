"""Contains utility functions for depth column entries."""


def parse_numeric_value(string_value: str) -> float | int:  # noqa: D103
    """Converts a string to a float."""
    # OCR sometimes tends to miss the decimal comma
    # parsed_text = re.sub(r"^-?([0-9]+)([0-9]{2})", r"\1.\2", string_value)
    if "." not in string_value:
        return abs(int(string_value))
    return abs(float(string_value))
