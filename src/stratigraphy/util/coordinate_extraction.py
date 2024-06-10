"""This module contains the CoordinateExtractor class."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

import fitz
import regex

from stratigraphy.util.extract_text import extract_text_lines
from stratigraphy.util.line import TextLine
from stratigraphy.util.util import read_params

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
    rect: fitz.Rect

    def __post_init__(self):
        # east always greater than north by definition. Irrespective of the leading 1 or 2
        if self.east.coordinate_value < self.north.coordinate_value:
            logger.info("Swapping coordinates.")
            east = self.north
            self.north = self.east
            self.east = east

    def __str__(self):
        return f"E: {self.east}, N: {self.north}"

    def to_json(self):
        return {
            "E": self.east.coordinate_value,
            "N": self.north.coordinate_value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
        }

    @abc.abstractmethod
    def is_valid(self):
        pass

    @staticmethod
    def from_values(east: int, north: int, rect: fitz.Rect) -> Coordinate | None:
        if 2e6 < east < 1e7:
            return LV95Coordinate(
                CoordinateEntry(coordinate_value=east), CoordinateEntry(coordinate_value=north), rect
            )
        elif east < 1e6:
            return LV03Coordinate(
                CoordinateEntry(coordinate_value=east), CoordinateEntry(coordinate_value=north), rect
            )
        else:
            logger.warning(f"Invalid coordinates format. Got E: {east}, N: {north}")
            return None

    @staticmethod
    def from_json(input: dict):
        return Coordinate.from_values(east=input["E"], north=input["N"], rect=fitz.Rect(input["rect"]))


@dataclass
class LV95Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV95 format."""

    east: CoordinateEntry
    north: CoordinateEntry
    rect: fitz.Rect

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten."""
        return 2324800 < self.east.coordinate_value < 2847500 and 1074000 < self.north.coordinate_value < 1302000


@dataclass
class LV03Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV03 format."""

    east: CoordinateEntry
    north: CoordinateEntry

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten.

        To account for uncertainties in the conversion of LV03 to LV95, we allow a margin of 2.
        """
        return 324798 < self.east.coordinate_value < 847502 and 73998 < self.north.coordinate_value < 302002


class CoordinateExtractor:
    """Extracts coordinates from a PDF document."""

    def __init__(self, document: fitz.Document):
        """Initializes the CoordinateExtractor object.

        Args:
            document (fitz.Document): A PDF document.
        """
        self.doc = document
        self.coordinate_keys = read_params("matching_params.yml")["coordinate_keys"]

    def get_coordinates_with_x_y_labels(self, lines: list[TextLine]) -> list[Coordinate]:
        """Find coordinates with explicit "X" and "Y" labels from the text lines.

        Args:
            lines (list[TextLine]): all the lines of text to search in

        Returns:
            list[Coordinate]: all found coordinates
        """
        # In this case, we can allow for some whitespace in between the numbers.
        # In some older borehole profile the OCR may recognize whitespace between two digits.
        pattern_x = regex.compile(r"X[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX)
        x_matches = CoordinateExtractor._match_text_with_rect(lines, pattern_x)

        pattern_y = regex.compile(r"Y[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX)
        y_matches = CoordinateExtractor._match_text_with_rect(lines, pattern_y)

        # We are only checking the 1st x-value with the 1st y-value, the 2nd x-value with the 2nd y-value, etc.
        # In some edge cases, the matched x_values and y-values might not be aligned / equal in number. However,
        # we ignore this for now, as almost always, the 1st x and y values are already the ones that we are looking
        # for.
        found_coordinates = []
        for x_match, y_match in zip(x_matches, y_matches, strict=False):
            rect = fitz.Rect()
            rect.include_rect(x_match[1])
            rect.include_rect(y_match[1])
            coordinates = Coordinate.from_values(
                east=int("".join(x_match[0].groups(default=""))),
                north=int("".join(y_match[0].groups(default=""))),
                rect=rect,
            )
            if coordinates is not None and coordinates.is_valid():
                found_coordinates.append(coordinates)
        return found_coordinates

    def find_coordinate_key(self, lines: list[TextLine], allowed_errors: int = 3) -> TextLine | None:  # noqa: E501
        """Finds the location of a coordinate key in a string of text.

        This is useful to reduce the text within which the coordinates are searched. If the text is too large
        false positive (found coordinates that are no coordinates) are more likely.

        The function allows for a certain number of errors in the key. Errors are defined as insertions, deletions
        or substitutions of characters (i.e. Levenshtein distance). For more information of how errors are defined see
        https://github.com/mrabarnett/mrab-regex?tab=readme-ov-file#approximate-fuzzy-matching-hg-issue-12-hg-issue-41-hg-issue-109.


        Args:
            lines (list[TextLine]): Arbitrary text lines to search in.
            allowed_errors (int, optional): The maximum number of errors (Levenshtein distance) to consider a key
                                            contained in text. Defaults to 3 (guestimation; no optimisation done yet).

        Returns:
            TextLine | None: The line of the coordinate key found in the text.
        """
        matches = []
        for key in self.coordinate_keys:
            pattern = regex.compile(r"\b(" + key + "){e<" + str(allowed_errors) + r"}\b", flags=regex.IGNORECASE)
            for line in lines:
                match = pattern.search(line.text)
                if match:
                    matches.append((line, sum(match.fuzzy_counts)))

        # if no match was found, return None
        if len(matches) == 0:
            return None

        best_match = min(matches, key=lambda x: x[1])
        return best_match[0]

    def get_coordinates_near_key(self, lines: list[TextLine], page_width: float) -> list[Coordinate]:
        """Find coordinates from text lines that are close to an explicit "coordinates" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page_width (float): the width of the current page (in points / PyMuPDF coordinates)

        Returns:
            list[Coordinate]: all found coordinates
        """
        coord_lines = self.get_coordinate_lines(lines, page_width)

        def preprocess(value: str) -> str:
            value = value.replace(",", ".")
            value = value.replace("'", ".")
            value = value.replace("o", "0")  # frequent ocr error
            value = value.replace("\n", " ")
            return value

        return self.get_coordinates_from_lines(coord_lines, preprocess)

    def get_coordinate_lines(self, lines: list[TextLine], page_width: float) -> list[TextLine]:
        """Returns the substring of a text that contains the coordinate information.

        Args:
            lines (list[TextLine]): The lines of text to search in.
            page_width (float): The width of the page (in points / PyMuPDF coordinates)

        Returns:
            list[TextLIne]: The lines of the text that are close to an identified coordinate key.
        """
        # find the key that indicates the coordinate information
        coordinate_key_line = self.find_coordinate_key(lines)
        if coordinate_key_line is None:
            return []

        key_rect = coordinate_key_line.rect
        # look for coordinate values to the right and/or immediately below the key
        coordinate_search_rect = fitz.Rect(key_rect.x0, key_rect.y0, page_width, key_rect.y1 + 3 * key_rect.height)
        return [line for line in lines if line.rect.intersects(coordinate_search_rect)]

    @staticmethod
    def get_coordinates_from_lines(lines: list[TextLine], preprocess=lambda x: x) -> list[Coordinate]:
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
            lines (list[TextLine]): Arbitrary string of text.
            preprocess: function that takes a string and returns a preprocessed string  # TODO add type

        Returns:
            list[Coordinate]: A list of potential coordinates
        """
        full_regex = regex.compile(
            r"(?:[XY][=:\s]{0,2})?" + COORDINATE_ENTRY_REGEX + r".{0,4}?[XY]?[=:\s]{0,2}" + COORDINATE_ENTRY_REGEX
        )
        potential_coordinates = [
            Coordinate.from_values(
                east=int("".join(match.groups(default="")[:3])),
                north=int("".join(match.groups(default="")[3:])),
                rect=rect,
            )
            for match, rect in CoordinateExtractor._match_text_with_rect(lines, full_regex, preprocess)
        ]
        return [
            coordinates for coordinates in potential_coordinates if coordinates is not None and coordinates.is_valid()
        ]

    @staticmethod
    def _match_text_with_rect(
        lines: list[TextLine], pattern: regex.Regex, preprocess=lambda x: x
    ) -> list[(regex.Match, fitz.Rect)]:
        full_text = ""
        lines_with_position = []
        for line in lines:
            preprocessed_text = preprocess(line.text)
            lines_with_position.append(
                {"line": line, "start": len(full_text), "end": len(full_text) + len(preprocessed_text)}
            )
            full_text += preprocessed_text + " "

        results = []
        for match in pattern.finditer(full_text):
            match_lines = [
                entry["line"]
                for entry in lines_with_position
                if entry["end"] >= match.start() and entry["start"] < match.end()
            ]

            rect = fitz.Rect()
            for line in match_lines:
                rect.include_rect(line.rect)
            results.append((match, rect))
        return results

    def extract_coordinates(self) -> Coordinate | None:
        """Extracts the coordinates from a string of text.

        Algorithm description:
            1. search for coordinates with explicit 'X' and 'Y' labels
            2. if that gives no results, search for coordinates close to an explicit "coordinates" label
            3. if that gives no results either, try to detect coordinates in the full text

        Returns:
            Coordinate | None: the extracted coordinates (if any)
        """
        for page in self.doc:
            lines = extract_text_lines(page)

            found_coordinates = (
                self.get_coordinates_with_x_y_labels(lines)
                or self.get_coordinates_near_key(lines, page.rect.width)
                or self.get_coordinates_from_lines(lines)
            )

            if len(found_coordinates) > 0:
                return found_coordinates[0]

        logger.info("No coordinates found in this borehole profile.")
