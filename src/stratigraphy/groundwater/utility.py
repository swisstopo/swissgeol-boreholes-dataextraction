"""Series of utility functions for groundwater stratigraphy."""

import regex


def extract_date(text: str) -> str | None:
    """Extract the date from a string."""
    date = None
    date_match = regex.search(r"(\d{1,2}\.\d{1,2}\.\d{2,4})", text)
    if date_match:
        date = date_match.group(1)
    return date


def extract_depth(text: str) -> float | None:
    """Extract the depth from a string.

    Args:
        text (str): The text to extract the depth from.

    Returns:
        float: The extracted depth.
    """
    depth_patterns = [
        r"([\d.]+)\s*m\s*u\.t\.",
        r"([\d.]+)\s*m\s*u\.t",
        r"(\d+.\d+)",
    ]

    depth = None
    corrected_text = correct_ocr_text(text).lower()
    for pattern in depth_patterns:
        depth_match = regex.search(pattern, corrected_text)
        if depth_match:
            depth = float(depth_match.group(1).replace(",", "."))
            break
    return depth


def extract_elevation(text: str) -> float | None:
    """Extract the elevation from a string.

    Args:
        text (str): The text to extract the elevation from.

    Returns:
        float: The extracted elevation.
    """
    elevation_patterns = [
        r"([\d.]+)\s*m\s*u\.m\.",
        r"([\d.]+)\s*m\s*ur.",
    ]

    elevation = None
    for pattern in elevation_patterns:
        elevation_match = regex.search(pattern, text.lower())
        if elevation_match:
            elevation = float(elevation_match.group(1).replace(",", "."))
            break
    return elevation


def correct_ocr_text(text):
    """Corrects common OCR errors in the text.

    Example: "1,48 8 m u.T." -> "1,48 m u.T."

    Args:
        text (str): the text to correct

    Returns:
        str: the corrected text
    """
    # Regex pattern to find a float number followed by a duplicate number and then some text
    pattern = r"(\b\d+\.\d+)\s*(\d*)\s*(m\s+u\.T\.)"

    # Search for the pattern in the text
    match = regex.search(pattern, text)

    if match:
        float_number = match.group(1)  # The valid float number
        rest_of_text = match.group(3)  # The remaining text, e.g., 'm u.T.'

        # If the duplicate exists and matches part of the float, remove it
        corrected_text = f"{float_number} {rest_of_text}"
    else:
        corrected_text = text  # Return original if no match

    return corrected_text
