"""This module contains the CoordinateExtractor class."""

from dataclasses import dataclass

import fitz
import regex


@dataclass
class CoordinateEntry:
    """Dataclass to represent a coordinate entry."""

    first_entry: int
    second_entry: int

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
        return Coordinate(
            CoordinateEntry(int(latitude[0]), int(latitude[1])), CoordinateEntry(int(longitude[0]), int(longitude[1]))
        )


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
            return None

        coord_start = text.find(key) + len(key)
        coord_end = coord_start + 100
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
        return regex.findall(r"X?[=:\s]?\d{3}[\.\s]\d{3}.*Y?[=:\s]?\d{3}[\.\s]\d{3}", text)

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
            return None

        # match the format X[=:\s]123[.\s]456 Y[=:\s]123[.\s]456
        try:
            coordinate_string = self.get_coordinates_text(coord_substring)[0]

        except IndexError:  # no coordinates found
            return None

        matches = regex.findall(r"\d{3}[\.\s]\d{3}", coordinate_string)
        lattitude1, lattitued2 = regex.findall(r"\d{3}", matches[0])
        longitude1, longitude2 = regex.findall(r"\d{3}", matches[1])

        latitude = CoordinateEntry(int(lattitude1), int(lattitued2))
        longitude = CoordinateEntry(int(longitude1), int(longitude2))

        return Coordinate(latitude, longitude)
