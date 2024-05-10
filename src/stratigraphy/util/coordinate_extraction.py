"""This module contains the CoordinateExtractor class."""

import re

import fitz
import regex


class CoordinateExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.coordinate_keys = ["Koordinaten", "Koordinate", "coordinates", "coordinate"]

    def find_coordinate_key(self, text: str, allowed_operations: int = 3) -> str:
        """Finds the location of a coordinate key in a string of text.

        Args:
            text (str): Arbitrary string of text.
            allowed_operations (int, optional): The maximum number of allowed operations to consider a key contained
                                                in text. Defaults to 3.

        Returns:
            str: The coordinate key found in the text.
        """
        matches = []
        for key in self.coordinate_keys:
            match = regex.search(r"\b(" + key + "){e<" + str(allowed_operations) + "}\s", text, flags=regex.IGNORECASE)
            if match:
                matches.append((match.group(), sum(match.fuzzy_counts)))

        # if no match was found, return None
        if matches == []:
            return None

        best_match = sorted(matches, key=lambda x: x[1], reverse=True)[0][0]

        return best_match

    def get_coordinate_substring(self, text: str) -> str:
        """Returns the substring of a text that contains the coordinate information.

        Args:
            text (str): Arbitrary string of text.

        Returns:
            str: The substring of the text that contains the coordinate information.
        """
        # find the key that indicates the coordinate information
        key = self.find_coordinate_key(text)
        # if no key was found, return None
        if key is None:
            return None

        coord_start = text.find(key) + len(key)
        coord_end = coord_start + 50
        return text[coord_start:coord_end]

    def extract_coordinates(self) -> list:
        """Extracts the coordinates from a string of text.

        Returns:
            list: A list of coordinates.
        """
        text = ""
        for page in self.doc:
            text += page.get_text()

        # get the substring that contains the coordinate information
        coord_substring = self.get_coordinate_substring(text)

        # if no coordinate information was found, return an empty list
        if coord_substring is None:
            return []

        # extract the coordinates
        coordinates = re.findall(r"[-+]?\d*\.\d+|\d+", coord_substring)
        return coordinates
