"""Module for data extraction from stratigraphy data files.

This module defines the DataExtractor class for extracting data from stratigraphy data files.
"""

import logging
from abc import ABC, abstractmethod

import fitz
import regex
from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)


class DataExtractor(ABC):
    """Abstract class for data extraction from stratigraphy data files.

    This class defines the interface for extracting data from stratigraphy data files. Subclasses must implement the
    extract_data method to define the data extraction logic.
    """

    doc: fitz.Document = None
    feature_keys: list[str] = None
    feature_name: str = None

    def __init__(self, document: fitz.Document):
        """Initializes the DataExtractor object.

        Args:
            document (fitz.Document): A PDF document.
            feature_name (str): The name of the feature to extract.
        """
        if not self.feature_name:
            raise ValueError("Feature name must be specified.")

        self.doc = document
        self.feature_keys = read_params("matching_params.yml")[f"{self.feature_name}_keys"]

    def find_feature_key(self, lines: list[TextLine], allowed_errors: int = 3) -> TextLine | None:  # noqa: E501
        """Finds the location of a feature key in a string of text.

        This is useful to reduce the text within which the feature is searched. If the text is too large
        false positive (found feature that is actually not the feature) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            TextLine | None: The line of the feature key found in the text.
        """
        matches = []
        for key in self.feature_keys:
            if len(key) < 5:
                # if the key is very short, do an exact match
                pattern = regex.compile(r"\b" + key + r"\b", flags=regex.IGNORECASE)
            else:
                pattern = regex.compile(r"\b(" + key + "){e<" + str(allowed_errors) + r"}\b", flags=regex.IGNORECASE)

            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.append((line, sum(match.fuzzy_counts)))

        # if no match was found, return None
        if len(matches) == 0:
            return None

        # Remove duplicates
        matches = list(dict.fromkeys(matches))

        if self.feature_name == "coordinate":
            # Return the best match (line only)
            best_match = min(matches, key=lambda x: x[1])
            return best_match[0]
        elif self.feature_name in ["elevation", "groundwater"]:
            # Sort the matches by their error counts (ascending order)
            matches.sort(key=lambda x: x[1])

            # Return the top three matches (lines only)
            return [match[0] for match in matches[:5]]
        else:
            raise ValueError(f"Feature name '{self.feature_name}' not supported.")

    @abstractmethod
    def get_feature_near_key(self, lines: list[TextLine], page: int, page_width: float):
        """Finds the location of a feature near a key in a string of text.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            page (int): The page number (1-based) of the PDF document.
            page_width (float): The width of the page in pixels.

        Returns:
            TextLine | None: The line of the feature found near the key in the text.
        """
        pass

    def extract_data(self) -> dict:
        """Extracts the feature information (e.g., groundwater, elevation, coordinates) from a borehole profile.

        Processes the borehole profile page by page and tries to find the feature key in the respective text of the
        page.
        """
        for page in self.doc:
            lines = extract_text_lines(page)
            page_number = page.number + 1  # page.number is 0-based

            found_feature_value = (
                self.get_feature_near_key(lines, page_number, page.rect.width)
                # or XXXX # Add other techniques here
            )

            if self.feature_name in ["elevation", "coordinate"]:
                if found_feature_value:
                    feature_value = getattr(found_feature_value, self.feature_name)
                    logger.info(f"Found {self.feature_name} on page {page_number}: {feature_value}")
                    return found_feature_value
            elif self.feature_name == "groundwater":
                if len(found_feature_value):
                    feature_output = ", ".join(
                        [str(getattr(entry, self.feature_name)) for entry in found_feature_value]
                    )
                    logger.info(f"Found {self.feature_name} on page {page_number}: {feature_output}")
                    return found_feature_value
            else:
                raise ValueError(f"Feature name '{self.feature_name}' not supported")

        logger.info(f"No {self.feature_name}  found in this borehole profile.")
        return None
