"""This module contains the DrillingMethodExtractor class."""

import logging

import fitz
import regex

from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)


class DrillingMethodExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.drilling_method_keys = read_params("matching_params.yml")["drilling_method_keys"]

    def find_drilling_method_key(self, lines: list[TextLine], allowed_errors: int = 3) -> TextLine | None:  # noqa: E501
        """Finds the location of a drilling method key in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            TextLine | None: The line of the drilling method key found in the text.
        """
        matches = []
        for key in self.drilling_method_keys:
            pattern = regex.compile(r"\b" + key + "{e<" + str(allowed_errors) + r"}\b", flags=regex.IGNORECASE)
            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.append((line, sum(match.fuzzy_counts)))

        # if no match was found, return None
        if len(matches) == 0:
            return None

        best_match = min(matches, key=lambda x: x[1])
        return best_match[0]

    def get_drilling_method_near_key(self, lines: list[TextLine], page: int, page_width: float) -> list[TextLine]:
        """Find coordinates from text lines that are close to an explicit "coordinates" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document
            page_width (float): the width of the current page (in points / PyMuPDF coordinates)

        Returns:
            list[Coordinate]: all found coordinates
        """
        # find the key that indicates the coordinate information
        coordinate_key_line = self.find_drilling_method_key(lines)
        if coordinate_key_line is None:
            return []

        # find the lines of the text that are close to an identified coordinate key.
        key_rect = coordinate_key_line.rect
        # look for coordinate values to the right and/or immediately below the key
        coordinate_search_rect = fitz.Rect(key_rect.x0, key_rect.y0, page_width, key_rect.y1 + 3 * key_rect.height)
        coord_lines = [line for line in lines if line.rect.intersects(coordinate_search_rect)]

        # def preprocess(value: str) -> str:
        #     value = value.replace(",", ".")
        #     value = value.replace("'", ".")
        #     value = value.replace("o", "0")  # frequent ocr error
        #     value = value.replace("\n", " ")
        #     return value

        # return self.get_coordinates_from_lines(coord_lines, page, preprocess)

        return coord_lines

    def extract_drilling_method(self) -> TextLine | None:
        """Extracts the coordinates from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.

        Algorithm description:
            1. search for coordinates with explicit 'X' and 'Y' labels
            2. if that gives no results, search for coordinates close to an explicit "coordinates" label
            3. if that gives no results either, try to detect coordinates in the full text

        Returns:
            Coordinate | None: the extracted coordinates (if any)
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # page.number is 0-based

            found_coordinates = (
                self.get_drilling_method_near_key(lines, page_number, page.rect.width)
                # or XXXX # Add other techniques here
            )

            if len(found_coordinates) > 0:
                return found_coordinates[0]

        logger.info("No coordinates found in this borehole profile.")
