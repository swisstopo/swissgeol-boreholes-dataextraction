"""Module for data extraction from stratigraphy data files.

This module defines the DataExtractor class for extracting data from stratigraphy data files.
"""

import logging
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import fitz
import regex
from stratigraphy.data_extractor.utility import get_lines_near_rect
from stratigraphy.lines.line import TextLine
from stratigraphy.util.util import read_params

logger = logging.getLogger(__name__)


class ExtractedFeature(metaclass=ABCMeta):
    """Class for extracted feature information."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Checks if the information is valid.

        Returns:
            bool: True if the information is valid, otherwise False.
        """
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> Self:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the information.

        Returns:
            Self: An instance of the class.
        """
        pass


T = TypeVar("T", bound=ExtractedFeature)


@dataclass
class FeatureOnPage(Generic[T]):
    """Class for an extracted feature, together with the page and where on that page the feature was extracted from."""

    feature: T
    rect: fitz.Rect  # The rectangle that contains the extracted information
    page: int  # The page number of the PDF document

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        result = self.feature.to_json()
        result.update(
            {
                "page": self.page if self.page else None,
                "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
            }
        )
        return result

    @classmethod
    def from_json(cls, data: dict, feature_cls: type[T]) -> Self:
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the feature on a page information.
            feature_cls (T): The extracted feature

        Returns:
            Self: The resulting FeatureOnPage object.
        """
        return cls(
            feature=feature_cls.from_json(data),
            page=data["page"],
            rect=fitz.Rect(data["rect"]),
        )


class DataExtractor(ABC):
    """Abstract class for data extraction from stratigraphy data files.

    This class defines the interface for extracting data from stratigraphy data files.
    """

    doc: fitz.Document = None
    feature_keys: list[str] = None
    feature_fp_keys: list[str] = None
    feature_name: str = None

    # How much to the left of a key do we look for the feature information, as a multiple of the key line width
    search_left_factor: float = 0
    # How much to the right of a key do we look for the feature information, as a multiple of the key line width
    search_right_factor: float = 0
    # How much below a key do we look for the feature information, as a multiple of the key line height
    search_below_factor: float = 0
    # How much above a key do we look for the feature information, as a multiple of the key line height
    search_above_factor: float = 0

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
        self.feature_fp_keys = read_params("matching_params.yml")[f"{self.feature_name}_fp_keys"] or []

    def preprocess(self, value: str) -> str:
        """Preprocesses the value before searching for the feature.

        Args:
            value (str): The value to preprocess.

        Returns:
            str: The preprocessed value.
        """
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
                if match and (not any(fp_key in line.text for fp_key in self.feature_fp_keys)):
                    # Check if there is a match and the matched string is not in the false positive list
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
        feature_lines = self.get_lines_near_rect(lines, key_rect)

        # Insert key_line first and remove duplicates
        feature_lines.insert(0, key_line)
        feature_lines = list(dict.fromkeys(feature_lines))

        # Sort by vertical distance between the top of the feature line and the top of key_line
        feature_lines_sorted = sorted(feature_lines, key=lambda line: abs(line.rect.y0 - key_line.rect.y0))

        return feature_lines_sorted

    def get_lines_near_rect(self, lines, rect: fitz.Rect) -> list[TextLine]:
        """Find the lines of the text that are close to a given rectangle.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            rect (fitz.Rect): The rectangle to search around.

        Returns:
            list[TextLine]: The lines close to the rectangle.
        """
        return get_lines_near_rect(
            self.search_left_factor,
            self.search_right_factor,
            self.search_above_factor,
            self.search_below_factor,
            lines,
            rect,
        )
