"""This module contains the CoordinateExtractor class."""

import logging
from dataclasses import dataclass

import fitz
import regex

logger = logging.getLogger(__name__)


@dataclass
class CoordinateEntry:
    """Dataclass to represent a coordinate entry."""

    first_entry: str
    second_entry: str

    def __repr__(self):
        return f"{self.first_entry}.{self.second_entry}"


@dataclass
class Coordinate:
    """Dataclass to represent a coordinate."""

    latitude: CoordinateEntry
    longitude: CoordinateEntry

    def __repr__(self):
        return f"Latitude: {self.latitude.first_entry}.{self.latitude.second_entry}, \
               Longitude: {self.longitude.first_entry}.{self.longitude.second_entry}"

    def to_json(self):
        return {
            "latitude": f"{self.latitude.first_entry}.{self.latitude.second_entry}",
            "longitude": f"{self.longitude.first_entry}.{self.longitude.second_entry}",
        }

    @staticmethod
    def from_json(input: dict):
        latitude = input["latitude"].split(".")
        longitude = input["longitude"].split(".")
        return Coordinate(CoordinateEntry(latitude[0], latitude[1]), CoordinateEntry(longitude[0], longitude[1]))


class CoordinateExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.coordinate_keys = ["Koordinaten", "Koordinate", "coordinates", "coordinate", "coordonnés"]

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
            return ""

        coord_start = text.find(key) + len(key)
        coord_end = coord_start + 100  # 100 seems to be enough to capture the coordinates;
        # and not too much to introduce random numbers
        substring = text[coord_start:coord_end]
        substring = substring.replace(",", ".")
        substring = substring.replace("'", ".")
        substring = substring.replace("o", "0")  # frequent ocr error
        substring = substring.replace("\n", " ")
        return substring

    @staticmethod
    def get_coordinates_text(text: str) -> list:
        """Matches the coordinates in a string of text.

        Args:
            text (str): Arbitrary string of text.

        Returns:
            list: A list of matched coordinates.
        """
        return regex.findall(r"X?[=:\s]?\d{3}[\.\s\']{0,2}\d{3}\.?\d?.*Y?[=:\s]?\d{3}[\.\s\']?\d{3}\.?\d?", text)

    def extract_coordinates(self) -> list:
        """Extracts the coordinates from a string of text.

        Returns:
            list: A list of coordinates.
        """
        text = ""
        for page in self.doc:
            text += page.get_text()
        text = text.replace("\n", " ")

        # try to get the text by including X and Y
        try:
            y_coordinate_string = regex.findall(r"Y[=:\s]{0,2}\d{3}[\.\s\']{0,2}\d{3}\.?\d?", text)
            x_coordinate_string = regex.findall(r"X[=:\s]{0,2}\d{3}[\.\s\']{0,2}\d{3}\.?\d?", text)
            coordinate_string = y_coordinate_string[0] + " / " + x_coordinate_string[0]
        except IndexError:  # no coordinates found
            try:
                # get the substring that contains the coordinate information
                # match the format X[=:\s]123[.\s]456 Y[=:\s]123[.\s]456
                coord_substring = self.get_coordinate_substring(text)
                coordinate_string = self.get_coordinates_text(coord_substring)[0]
            except IndexError:
                # if that doesnt work, try to directly detect coordinates in the text
                try:
                    coordinate_string = self.get_coordinates_text(text)[0]
                except IndexError:
                    return None

        matches = regex.findall(r"\d{3}[\.\s']{1,2}\d{3}", coordinate_string)
        if len(matches) >= 2:
            lattitude1, lattitude2 = regex.findall(r"\d{3}", matches[0])
            longitude1, longitude2 = regex.findall(r"\d{3}", matches[1])
        else:
            try:
                matches = regex.findall(r"\d{6}", coordinate_string)
                lattitude1, lattitude2 = matches[0][:3], matches[0][3:]
                longitude1, longitude2 = matches[1][:3], matches[1][3:]
            except IndexError:
                logger.warning(f"Could not extract coordinates from: {coordinate_string}")
                return None

        latitude = CoordinateEntry(lattitude1, lattitude2)
        longitude = CoordinateEntry(longitude1, longitude2)

        return Coordinate(latitude, longitude)
