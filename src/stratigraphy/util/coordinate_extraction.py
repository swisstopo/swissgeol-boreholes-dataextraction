"""This module contains the CoordinateExtractor class."""

import abc
import logging
from dataclasses import dataclass

import fitz
import regex

logger = logging.getLogger(__name__)

COORDINATE_ENTRY_REGEX = r"(?:[12][\.\s']{0,2})?\d{3}[\.\s']{0,2}\d{3}\.?\d?"


@dataclass
class CoordinateEntry:
    """Dataclass to represent a coordinate entry."""

    coordinate_value: int | None = None

    @staticmethod
    def create_from_string(first_entry: str, second_entry: str = None):
        coordinate_value = int(first_entry) if second_entry is None else int(first_entry + second_entry)
        return CoordinateEntry(coordinate_value)

    def __repr__(self):
        if self.coordinate_value > 1e5:
            return f"{self.coordinate_value:,}".replace(",", "'")
        else:  # Fix for LV03 coordinates with leading 0
            return f"{self.coordinate_value:07,}".replace(",", "'")


@dataclass
class Coordinate(metaclass=abc.ABCMeta):
    """Abstract class for coordinates."""

    east: CoordinateEntry
    north: CoordinateEntry

    def __post_init__(self):
        # east always greater than north by definition. Irrespective of the leading 1 or 2
        if self.east.coordinate_value < self.north.coordinate_value:
            logger.info("Swapping coordinates.")
            east = self.north
            self.north = self.east
            self.east = east

    @abc.abstractmethod
    def __repr__(self):  # noqa: D105
        pass

    @abc.abstractmethod
    def to_json(self):
        pass

    @abc.abstractmethod
    def is_valid(self):
        pass

    @staticmethod
    def from_json(input: dict):
        east = input["E"]
        north = input["N"]
        if east > 2e6 and east < 1e7:
            return LV95Coordinate(CoordinateEntry(coordinate_value=east), CoordinateEntry(coordinate_value=north))
        elif east < 1e6:
            return LV03Coordinate(CoordinateEntry(coordinate_value=east), CoordinateEntry(coordinate_value=north))
        else:
            logger.warning(f"Invalid coordinates format. Got E: {east}, N: {north}")
            return None


@dataclass
class LV95Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV95 format."""

    east: CoordinateEntry
    north: CoordinateEntry

    def __repr__(self):
        return f"E: {self.east}, " f"N: {self.north}"

    def to_json(self):
        return {
            "E": self.east.coordinate_value,
            "N": self.north.coordinate_value,
        }

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten."""
        return (
            self.east.coordinate_value > 2324800
            and self.east.coordinate_value < 2847500
            and self.north.coordinate_value > 1074000
            and self.north.coordinate_value < 1302000
        )


@dataclass
class LV03Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV03 format."""

    east: CoordinateEntry
    north: CoordinateEntry

    def __repr__(self):
        return f"E: {self.east}, " f"N: {self.north}"

    def to_json(self):
        return {
            "E": self.east.coordinate_value,
            "N": self.north.coordinate_value,
        }

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten.

        To account for uncertainties in the conversion of LV03 to LV95, we allow a margin of 2.
        """
        return (
            self.east.coordinate_value > 324798
            and self.east.coordinate_value < 847502
            and self.north.coordinate_value > 73998
            and self.north.coordinate_value < 302002
        )


class CoordinateExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.coordinate_keys = ["Koordinaten", "Koordinate", "coordinates", "coordinate", "coordonnés", "coordonnes"]
        # TODO: extend coordinate keys with other languages

    def find_coordinate_key(self, text: str, allowed_errors: int = 3) -> str:  # noqa: E501
        """Finds the location of a coordinate key in a string of text.

        This is is useful to reduce the text within which the coordinates are searched. If the text is too large
        false positive (found coordinates that are no coordinates) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            text (str): Arbitrary string of text.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            str: The coordinate key found in the text.
        """
        matches = []
        for key in self.coordinate_keys:
            match = regex.search(r"\b(" + key + "){e<" + str(allowed_errors) + r"}\s", text, flags=regex.IGNORECASE)
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
        r"""Matches the coordinates in a string of text.

        The full regular expressions query is:
        "[XY]?[=:\s]{0,2}(?:[12][\.\s']{0,2})?\d{3}[\.\s']{0,2}\d{3}\.?\d?.*[XY]?[=:\s]{0,2}(?:[12][\.\s']{0,2})?\d{3}[\.\s']?\d{3}\.?\d?"
        Query explanation:
            - [XY]?: This matches an optional 'X' or 'Y'. The ? makes the preceding character optional.
            - [=:\s]{0,2}: This matches zero to two occurrences of either an equals sign, a colon, or a whitespace
              character.
            - (?:[12][\.\s']{0,2})?: This is a non-capturing group (indicated by ?:), which means it groups the
              enclosed characters but does not create a backreference. It matches an optional '1' or '2' followed
              by zero to two occurrences of a period, space, or single quote.
            - \d{3}: This matches exactly three digits.
            - [\.\s']{0,2}: This matches zero to two occurrences of a period, space, or single quote.
            - \d{3}: This again matches exactly three digits.
            - \.?\d?: This matches an optional period followed by an optional digit.
            - .*: This matches any number of any characters, except newline.

            The second half of the regular expression repeats the pattern, allowing it to match a pair of coordinates
            in the format 'X=123.456 Y=123.456', with some flexibility for variations in the format. For example, it
            can also match 'X:123.456, Y:123.456', 'X 123 456 Y 123 456', and so on.

        Args:
            text (str): Arbitrary string of text.

        Returns:
            list: A list of matched coordinates.
        """
        return regex.findall(
            r"[XY]?[=:\s]{0,2}" + COORDINATE_ENTRY_REGEX + r".{0,4}[XY]?[=:\s]{0,2}" + COORDINATE_ENTRY_REGEX,
            text,
        )

    def extract_coordinates(self) -> list:
        """Extracts the coordinates from a string of text.

        Algorithm description:
            - Try to find the coordinate key in the text.
            - If the key is found, extract the substring that contains the coordinates.
            - If the key is not found, try to directly detect coordinates in the text.

        Returns:
            list: A list of coordinates.
        """
        text = ""
        for page in self.doc:
            text += page.get_text()
        text = text.replace("\n", " ")

        # try to get the text by including X and Y
        try:
            y_coordinate_string = regex.findall(r"Y[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, text)
            x_coordinate_string = regex.findall(r"X[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, text)
            coordinate_string = (
                y_coordinate_string[0].replace(" ", "") + " / " + x_coordinate_string[0].replace(" ", "")
            )
            # if we have a 'Y' and 'X' coordinate, we can allow for some whitespace in between the numbers.
            # In some older borehole profile the OCR may recognize whitespace between two digits.
        except IndexError:  # no coordinates found
            try:
                # get the substring that contains the coordinate information
                coord_substring = self.get_coordinate_substring(text)
                coordinate_string = self.get_coordinates_text(coord_substring)[0]
            except IndexError:
                # if that doesnt work, try to directly detect coordinates in the text
                try:
                    coordinate_string = self.get_coordinates_text(text)[0]
                except IndexError:
                    logger.info("No coordinates found in this borehole profile.")
                    return None
        matches = regex.findall(r"(?:[12][\.\s']{0,2})?\d{3}[\.\s']{1,2}\d{3}", coordinate_string)
        if len(matches) >= 2:
            east1, east2 = regex.findall(r"(?:2[\.\s']{0,2})?\d{3}", matches[0])
            north1, north2 = regex.findall(r"(?:1[\.\s']{0,2})?\d{3}", matches[1])
            if len(east1) > 3 and len(north1) > 3:
                # remove all characters that are not number from east1 and north1
                # this is necessary because in some cases there is a separator after the
                # leading 2 or 1.
                east1 = regex.sub(r"\D", "", east1)
                north1 = regex.sub(r"\D", "", north1)
                coordinate = LV95Coordinate(
                    CoordinateEntry.create_from_string(east1, east2),
                    CoordinateEntry.create_from_string(north1, north2),
                )

            else:
                # in some strange cases we recognize either east or north with 4 digits
                # in these case we just truncate to the required 3 digits
                east1 = east1[-3:]
                north1 = north1[-3:]
                coordinate = LV03Coordinate(
                    CoordinateEntry.create_from_string(east1, east2),
                    CoordinateEntry.create_from_string(north1, north2),
                )
            if coordinate.is_valid():
                return coordinate
            else:
                return None

        else:
            try:
                matches = regex.findall(r"[12]?\d{6}", coordinate_string)
                if len(matches[0]) == 6:  # we expect matches[0] and matches[1] to have the same length
                    coordinate = LV03Coordinate(
                        CoordinateEntry.create_from_string(matches[0]), CoordinateEntry.create_from_string(matches[1])
                    )

                if len(matches[0]) == 7:
                    coordinate = LV95Coordinate(
                        CoordinateEntry.create_from_string(matches[0]), CoordinateEntry.create_from_string(matches[1])
                    )

                if isinstance(coordinate, Coordinate) and coordinate.is_valid():
                    return coordinate

                return None

            except IndexError:
                logger.warning(f"Could not extract coordinates from: {coordinate_string}")
                return None
