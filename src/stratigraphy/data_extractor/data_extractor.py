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

    def find_feature_key(self, lines: list[TextLine], allowed_errors: int = 3) -> list[TextLine] | None:  # noqa: E501
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
        matches = set()
        for key in self.feature_keys:
            if len(key) < 5:
                # if the key is very short, do an exact match
                pattern = regex.compile(r"\b" + key + r"\b", flags=regex.IGNORECASE)
            else:
                pattern = regex.compile(r"\b" + key + "{e<" + str(allowed_errors) + r"}\b", flags=regex.IGNORECASE)

            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.add(line)

        return list(matches)

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

    @abstractmethod
    def extract_data(self) -> ExtractedFeature | list[ExtractedFeature] | None:
        """Extracts the feature information (e.g., groundwater, elevation, coordinates) from a borehole profile.

        Processes the borehole profile page by page and tries to find the feature key in the respective text of the
        page.
        """
        pass
