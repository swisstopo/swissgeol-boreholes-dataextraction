"""Series of utility functions for groundwater stratigraphy."""

from datetime import date, datetime

import regex


def extract_date(text: str) -> tuple[date | None, str | None]:
    """Extract the date from a string in the format dd.mm.yyyy or dd.mm.yy."""
    date_match = regex.search(r"(\d{1,2})[\s\.]+(\d{1,2})[\s\.]+(\d{2,4})", text)

    if not date_match:
        return None, None
    original_date_str = date_match.group(0)
    cleaned_date_str = f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"
    date_format = "%d.%m.%y" if len(date_match.group(3)) == 2 else "%d.%m.%Y"
    # Validate date before parsing
    if not is_valid_date(cleaned_date_str, date_format):
        return None, None
    dt = datetime.strptime(cleaned_date_str, date_format)
    if dt > datetime.now():
        dt = dt.replace(year=dt.year - 100)

    return dt.date(), original_date_str


def is_valid_date(date_str: str, date_format: str) -> bool:
    """Check if a date string is valid for a given format."""
    try:
        datetime.strptime(date_str, date_format)  # Try parsing the date
        return True
    except ValueError:
        return False


def extract_depth(text: str, max_depth: int) -> float | None:
    """Extract the depth from a string.

    Args:
        text (str): The text to extract the depth from.
        max_depth (int): The maximum depth allowed.

    Returns:
        float: The extracted depth.
    """
    depth_patterns = [
        r"([\d.]+)\s*m\s*u\.t\.",  # e.g. "5.13 m u.T."
        r"([\d.]+)\s*m\s*u\.t",
        r"(\d+\.\d+)",
    ]

    depth = None
    corrected_text = correct_ocr_text(text).lower()
    for pattern in depth_patterns:
        depth_match = regex.search(pattern, corrected_text)
        try:
            if depth_match:
                depth = float(depth_match.group(1).replace(",", "."))
                if depth > max_depth:
                    # If the extracted depth is greater than the max depth, set it to None and continue searching.
                    depth = None
                else:
                    break
        except ValueError:
            continue
    return depth


def extract_elevation(text: str) -> float | None:
    """Extract the elevation from a string.

    Args:
        text (str): The text to extract the elevation from.

    Returns:
        float: The extracted elevation.
    """
    elevation_patterns = [
        r"(\d+(\.\d+)?)\s*m\s*u\.m\.",
        r"(\d+(\.\d+)?)\s*m\s*ur.",
        r"(\d{3,}\.\d{1,2})(?!\d)",  # Matches a float number with less than 2 digits after the decimal point
        r"(\d{3,})\s*m",
    ]

    elevation = None
    for pattern in elevation_patterns:
        elevation_match = regex.search(pattern, text.lower().replace(", ", ",").replace(". ", "."))
        if elevation_match:
            elevation = float(elevation_match.group(1).replace(" ", "").replace(",", "."))
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
