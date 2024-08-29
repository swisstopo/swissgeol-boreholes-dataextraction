"""Helper functions for working with coordinates."""

import re

from stratigraphy.util.coordinate_extraction import COORDINATE_ENTRY_REGEX


def get_coordinate_numbers_from_string(string: str) -> tuple[float]:
    """Extract the first two numbers from a string.

    Supports various notation of coordinates.

    Supported coordinate formats are:
        - "2'456'435"
        - "2456435"
        - "2.456.435"
        - "2456435"
        - "2,456,435"
        - "456'435"
        - etc.

    Args:
        string (str): The string to extract the number from.

    Returns:
        tuple[float]: The extracted numbers.
    """
    numbers = re.findall(COORDINATE_ENTRY_REGEX, string)
    if len(numbers) == 2:
        return int("".join(numbers[0])), int("".join(numbers[1]))
    else:
        return tuple()
