"""This module contains the CoordinateExtractor class."""

import abc
import logging
from dataclasses import dataclass

import fitz
import regex

logger = logging.getLogger(__name__)

COORDINATE_ENTRY_REGEX = r"(?:([12])[\.\s'‘’]{0,2})?(\d{3})[\.\s'‘’]{0,2}(\d{3})\.?\d?"


@dataclass
class CoordinateEntry:
    """Dataclass to represent a coordinate entry."""

    coordinate_value: int

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
        self.coordinate_keys = [
            "Koordinaten",
            "Koordinate",
            "Koord.",
            "coordinates",
            "coordinate",
            "coordonnés",
            "coordonnes",
        ]
        # TODO: extend coordinate keys with other languages

    def find_coordinate_key(self, text: str, allowed_errors: int = 3) -> str | None:  # noqa: E501
        """Finds the location of a coordinate key in a string of text.

        This is useful to reduce the text within which the coordinates are searched. If the text is too large
        false positive (found coordinates that are no coordinates) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            text (str): Arbitrary string of text.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            str | None: The coordinate key found in the text.
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
    def get_coordinate_pairs(text: str) -> list:
        r"""Matches the coordinates in a string of text.

        The query searches for a pair of coordinates of 6 or 7 digits, respectively. The pair of coordinates
        must at most be separated by 4 characters. The regular expression is designed to match a wide range of
        coordinate formats for the Swiss coordinate systems LV03 and LV95, including 'X=123.456 Y=123.456',
        'X:123.456, Y:123.456', 'X 123 456 Y 123 456', whereby the X and Y are optional.

        The full regular expressions query is:
        "[XY]?[=:\s]{0,2}(?:([12])[\.\s'‘’]{0,2})?(\d{3})[\.\s'‘’]{0,2}(\d{3})\.?\d?.{0,4}?[XY]?[=:\s]{0,2}(?:([12])[\.\s'‘’]{0,2})?(\d{3})[\.\s'‘’]{0,2}(\d{3})\.?\d?"
        Query explanation:
            - [XY]?: This matches an optional 'X' or 'Y'. The ? makes the preceding character optional.
            - [=:\s]{0,2}: This matches zero to two occurrences of either an equals sign, a colon, or a whitespace
              character.
            - (?:([12])[\.\s'‘’]{0,2})?: This is a non-capturing group (indicated by ?:), which means it groups the
              enclosed characters but does not create a backreference. It matches an optional '1' or '2' (which is
              captured in a group) followed by zero to two occurrences of a period, space, or single quote.
            - \d{3}: This matches exactly three digits.
            - [\.\s'‘’]{0,2}: This matches zero to two occurrences of a period, space, or single quote.
            - \d{3}: This again matches exactly three digits.
            - \.?\d?: This matches an optional period followed by an optional digit.
            - .{0,4}?: This matches up to four occurrences of any characters, except newline.

            The second half of the regular expression repeats the pattern, allowing it to match a pair of coordinates
            in the format 'X=123.456 Y=123.456', with some flexibility for variations in the format. For example, it
            can also match 'X:123.456, Y:123.456', 'X 123 456 Y 123 456', and so on.

        Args:
            text (str): Arbitrary string of text.

        Returns:
            list: A list of matched coordinate pairs, e.g. (2600000, 1200000)
        """
        full_regex = r"[XY]?[=:\s]{0,2}" + COORDINATE_ENTRY_REGEX + r".{0,4}?[XY]?[=:\s]{0,2}" + COORDINATE_ENTRY_REGEX
        return [
            (int("".join(groups[:3])), int("".join(groups[3:])))
            for groups in regex.findall(
                full_regex,
                text,
            )
        ]

    def extract_coordinates(self) -> Coordinate | None:
        """Extracts the coordinates from a string of text.

        Algorithm description:
            - Try to find the coordinate key in the text.
            - If the key is found, extract the substring that contains the coordinates.
            - If the key is not found, try to directly detect coordinates in the text.

        Returns:
            Coordinate | None: the extracted coordinates (if any)
        """
        text = ""
        for page in self.doc:
            text += page.get_text()
        text = text.replace("\n", " ")

        # try to get the text by including X and Y
        try:
            x_values = [
                int("".join(groups)) for groups in regex.findall(r"X[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, text)
            ]
            y_values = [
                int("".join(groups)) for groups in regex.findall(r"Y[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, text)
            ]

            coordinate_values = [(x_values[0], y_values[0])]
            # if we have a 'Y' and 'X' coordinate, we can allow for some whitespace in between the numbers.
            # In some older borehole profile the OCR may recognize whitespace between two digits.
        except IndexError:  # no coordinates found
            # get the substring that contains the coordinate information
            coord_substring = self.get_coordinate_substring(text)
            coordinate_values = self.get_coordinate_pairs(coord_substring)
            if len(coordinate_values) == 0:
                # if that doesn't work, try to directly detect coordinates in the text
                coordinate_values = self.get_coordinate_pairs(text)

        if len(coordinate_values) == 0:
            logger.info("No coordinates found in this borehole profile.")
            return None

        for east, north in coordinate_values:
            if east > 1e6 and north > 1e6:
                coordinate = LV95Coordinate(
                    CoordinateEntry(east),
                    CoordinateEntry(north),
                )
            else:
                coordinate = LV03Coordinate(
                    CoordinateEntry(east),
                    CoordinateEntry(north),
                )
            if coordinate.is_valid():
                return coordinate

        logger.warning(f"Could not extract valid coordinates from {coordinate_values}")
