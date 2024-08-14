"""Series of utility functions for groundwater stratigraphy."""

import regex


def extract_date(text: str) -> str | None:
    """Extract the date from a string."""
    date = None
    date_match = regex.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if date_match:
        date = date_match.group(1)
    else:
        # Try to match a date with a two-digit year
        date_match = regex.search(r"(\d{2}\.\d{2}\.\d{2})", text)
        if date_match:
            date = date_match.group(1)
        else:
            # Try to match a date with a one-digit month: 14.6.75
            date_match = regex.search(r"(\d{2}\.\d{1}\.\d{2})", text)
            if date_match:
                date = date_match.group(1)
            else:
                # Try to match a date with a four-digit year and one-digit month: 14.6.1975
                date_match = regex.search(r"(\d{2}\.\d{1}\.\d{4})", text)
                if date_match:
                    date = date_match.group(1)
                else:
                    # Try to match a date with a four-digit year and two-digit month: 14.06.1975
                    date_match = regex.search(r"(\d{1}\.\d{1}\.\d{2})", text)
                    if date_match:
                        date = date_match.group(1)
                    else:
                        # Try to match a date with a four-digit year and two-digit month: 14.06.1975
                        date_match = regex.search(r"(\d{1}\.\d{2}\.\d{4})", text)
                        if date_match:
                            date = date_match.group(1)
                        else:
                            # Try to match a date with a four-digit year and two-digit month: 14.06.1975
                            date_match = regex.search(r"(\d{1}\.\d{1}\.\d{4})", text)
                            if date_match:
                                date = date_match.group(1)
                            else:
                                # Try to match a date with a four-digit year and two-digit month: 14.06.1975
                                date_match = regex.search(r"(\d{1}\.\d{2}\.\d{2})", text)
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
    depth = None
    corrected_text = correct_ocr_text(text).lower()
    depth_match = regex.search(r"([\d.]+)\s*m\s*u\.t\.", corrected_text)
    if depth is None:
        # Only match depth if it has not been found yet
        if depth_match:
            depth = float(depth_match.group(1).replace(",", "."))
        else:
            depth_match = regex.search(r"([\d.]+)\s*m\s*u\.t", corrected_text)
            if depth_match:
                depth = float(depth_match.group(1).replace(",", "."))
            else:
                # Try to match a depth with a comma as decimal separator
                depth_match = regex.search(r"(\d+.\d+)", corrected_text)
                if depth_match:
                    depth = float(depth_match.group(1).replace(",", "."))
    return depth


def extract_elevation(text: str) -> float | None:
    """Extract the elevation from a string.

    Args:
        text (str): The text to extract the elevation from.

    Returns:
        float: The extracted elevation.
    """
    elevation = None
    elevation_match = regex.search(r"([\d.]+)\s*m\s*u\.m\.", text.lower())
    if elevation_match:
        elevation = float(elevation_match.group(1).replace(",", "."))
    else:
        elevation_match = regex.search(r"([\d.]+)\s*m\s*ur.", text.lower())
        if elevation_match:
            elevation = float(elevation_match.group(1).replace(",", "."))

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
