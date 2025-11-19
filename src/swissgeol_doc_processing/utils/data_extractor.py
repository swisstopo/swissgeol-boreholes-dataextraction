"""Module for data extraction from stratigraphy data files.

This module defines the DataExtractor class for extracting data from stratigraphy data files.
"""

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import pymupdf
import regex

from swissgeol_doc_processing.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from swissgeol_doc_processing.text.textline import TextLine

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFeature(metaclass=ABCMeta):
    """Class for extracted feature information."""

    is_correct = None

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


class FeatureOnPage(Generic[T], RectWithPageMixin):
    """Class for an extracted feature, together with the page and where on that page the feature was extracted from."""

    def __init__(self, feature: T, rect: pymupdf.Rect, page: int):
        self.feature = feature
        self.rect_with_page = RectWithPage(rect, page)

    def __repr__(self):
        return f"{self.feature}"

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        result = self.feature.to_json()
        result.update(
            {
                "page": self.page_number if self.page_number else None,
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
            rect=pymupdf.Rect(data["rect"]),
        )


class DataExtractor:
    """Abstract class for data extraction from stratigraphy data files.

    This class defines the interface for extracting data from stratigraphy data files.
    """

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

    def __init__(self, language: str, matching_params: dict):
        """Initializes the DataExtractor object.

        Args:
            language (str): the language of the document.
            matching_params (dict): The matching parameters.
        """
        if not self.feature_name:
            raise ValueError("Feature name must be specified.")

        self.feature_keys = matching_params[f"{self.feature_name}_keys"][language]
        self.feature_fp_keys = matching_params.get(f"{self.feature_name}_fp_keys", {}).get(language, [])

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
        feature_lines = self.get_axis_aligned_lines(lines, key_rect)

        # Insert key_line first and remove duplicates
        feature_lines.insert(0, key_line)
        feature_lines = list(dict.fromkeys(feature_lines))

        # Sort by
        # - vertical distance between the top of the feature line and the top of key_line
        # - horizontal position (left-first) for lines with identical vertical position
        feature_lines_sorted = sorted(
            feature_lines, key=lambda line: (abs(line.rect.y0 - key_line.rect.y0), line.rect.x0)
        )

        return feature_lines_sorted

    def get_axis_aligned_lines(self, lines: list[TextLine], rect: pymupdf.Rect) -> list[TextLine]:
        """Find the lines of text that are horizontally and vertically close to a given rectangle.

         Lines that are found both horizontally and vertically are included only once.

        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            rect (pymupdf.Rect): The rectangle to search around.

        Returns:
            list[TextLine]: A combined list of lines close to the rectangle within the horizontal
                            (left/right) and vertical (above/below) regions, with intersection included only once.
        """
        # Horizontal rectangle (left-right limits)
        horizontal_rect = pymupdf.Rect(
            rect.x0 - self.search_left_factor * rect.width,
            rect.y0,
            rect.x1 + self.search_right_factor * rect.width,
            rect.y1,
        )

        # Vertical rectangle (above-below limits)
        vertical_rect = pymupdf.Rect(
            rect.x0,
            rect.y0 - self.search_above_factor * rect.height,
            rect.x1,
            rect.y1 + self.search_below_factor * rect.height,
        )

        horizontal_lines = {line for line in lines if line.rect.intersects(horizontal_rect)}
        vertical_lines = {line for line in lines if line.rect.intersects(vertical_rect)}

        feature_lines = horizontal_lines | vertical_lines

        return list(feature_lines)
