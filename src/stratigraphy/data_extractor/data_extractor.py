"""Module for data extraction from stratigraphy data files.

This module defines the DataExtractor class for extracting data from stratigraphy data files.
"""

import logging
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass

import fitz
import regex
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ExtractedFeature(metaclass=ABCMeta):
    """Class for extracted feature information."""

    rect: fitz.Rect  # The rectangle that contains the extracted information
    page: int  # The page number of the PDF document

    @abstractmethod
    def is_valid(self) -> bool:
        """Checks if the information is valid.

        Returns:
            bool: True if the information is valid, otherwise False.
        """
        pass


class DataExtractor(ABC):
    """Abstract class for data extraction from stratigraphy data files.

    This class defines the interface for extracting data from stratigraphy data files.
    """

    feature_keys: list[str] = None
    feature_name: str = None

    # How much to the left of a key do we look for the feature information, as a multiple of the key line width
    search_left_factor: float = 0
    # How much to the right of a key do we look for the feature information, as a multiple of the key line width
    search_right_factor: float = 0
    # How much below a key do we look for the feature information, as a multiple of the key line height
    search_below_factor: float = 0

    preprocess_replacements: dict[str, str] = {}

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

    def preprocess(self, value: str) -> str:
        for old, new in self.preprocess_replacements.items():
            value = value.replace(old, new)
        return value

    def find_feature_key(self, lines: list[TextLine], allowed_error_rate: float = 0.2) -> list[TextLine]:  # noqa: E501
        """Finds the location of a feature key in a string of text.

        This is useful to reduce the text within which the feature is searched. If the text is too large
        false positive (found feature that is actually not the feature) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_error_rate (float, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                                  contained in text, as a percentage of the key length. Defaults to 0.2
                                                  (guestimation; no optimisation done yet).

        Returns:
            list[TextLine]: The lines of the feature key found in the text.
        """
        matches = set()

        for key in self.feature_keys:
            allowed_errors = int(len(key) * allowed_error_rate)
            if len(key) < 5:
                # If the key is very short, do an exact match
                pattern = regex.compile(r"(\b" + regex.escape(key) + r"\b)", flags=regex.IGNORECASE)
            else:
                # Allow for a certain number of errors in longer keys
                pattern = regex.compile(
                    r"(\b" + regex.escape(key) + r"\b){e<=" + str(allowed_errors) + r"}", flags=regex.IGNORECASE
                )

            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.add(line)

        return list(matches)

    def get_lines_near_key(self, lines, key_line: TextLine) -> list[TextLine]:
        """Find the lines of the text that are close to an identified key.

        The line of the identified key is always returned as the first item in the list.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            key_line (TextLine): The line of the identified key.

        Returns:
            list[TextLine]: The lines close to the key.
        """
        key_rect = key_line.rect
        elevation_search_rect = fitz.Rect(
            key_rect.x0 - self.search_left_factor * key_rect.width,
            key_rect.y0,
            key_rect.x1 + self.search_right_factor * key_rect.width,
            key_rect.y1 + self.search_below_factor * key_rect.height,
        )
        feature_lines = [line for line in lines if line.rect.intersects(elevation_search_rect)]

        # makes sure the line with the key is included first in the extracted information and the duplicate removed
        feature_lines.insert(0, key_line)
        return list(dict.fromkeys(feature_lines))
