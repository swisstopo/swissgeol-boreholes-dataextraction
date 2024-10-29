"""This module contains the CoordinateExtractor class."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

import fitz
import regex
from stratigraphy.data_extractor.data_extractor import DataExtractor, ExtractedFeature
from stratigraphy.lines.line import TextLine
from stratigraphy.text.extract_text import extract_text_lines_from_bbox

logger = logging.getLogger(__name__)

COORDINATE_ENTRY_REGEX = r"(?:([12])[\.\s'‘’]{0,2})?(\d{3})[\.\s'‘’]{0,2}(\d{3})(?:\.(\d{1,}))?"


@dataclass(kw_only=True)
class CoordinateEntry:
    """Dataclass to represent a coordinate entry."""

    coordinate_value: float

    def __repr__(self):
        if self.coordinate_value > 1e5:
            return f"{self.coordinate_value:,}".replace(",", "'")
        else:  # Fix for LV03 coordinates with leading 0
            return f"{self.coordinate_value:07,}".replace(",", "'")


@dataclass(kw_only=True)
class Coordinate(ExtractedFeature):
    """Abstract class for coordinates."""

    east: CoordinateEntry
    north: CoordinateEntry

    # TODO remove after refactoring to use FeatureOnPage also for coordinates
    rect: fitz.Rect  # The rectangle that contains the extracted information
    page: int  # The page number of the PDF document

    def __post_init__(self):
        # east always greater than north by definition. Irrespective of the leading 1 or 2
        if self.east.coordinate_value < self.north.coordinate_value:
            logger.info("Swapping coordinates.")
            self.north, self.east = self.east, self.north

    def __str__(self):
        return f"E: {self.east.coordinate_value}, N: {self.north.coordinate_value}"

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "E": self.east.coordinate_value,
            "N": self.north.coordinate_value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.page,
        }

    @abc.abstractmethod
    def is_valid(self):
        pass

    @staticmethod
    def from_values(east: float, north: float, rect: fitz.Rect, page: int) -> Coordinate | None:
        """Creates a Coordinate object from the given values.

        Args:
            east (float): The east coordinate value.
            north (float): The north coordinate value.
            rect (fitz.Rect): The rectangle that contains the extracted information.
            page (int): The page number of the PDF document.

        Returns:
            Coordinate | None: The coordinate object.
        """
        if 1e6 < east < 1e7:
            return LV95Coordinate(
                east=CoordinateEntry(coordinate_value=east),
                north=CoordinateEntry(coordinate_value=north),
                rect=rect,
                page=page,
            )
        elif east < 1e6:
            return LV03Coordinate(
                east=CoordinateEntry(coordinate_value=east),
                north=CoordinateEntry(coordinate_value=north),
                rect=rect,
                page=page,
            )
        else:
            logger.warning("Invalid coordinates format. Got E: %s, N: %s", east, north)
            return None

    @classmethod
    def from_json(cls, input: dict) -> Coordinate:
        """Converts a dictionary to a Coordinate object.

        Args:
            input (dict): A dictionary containing the coordinate information.

        Returns:
            Coordinate: The coordinate object.
        """
        return Coordinate.from_values(
            east=input["E"], north=input["N"], rect=fitz.Rect(input["rect"]), page=input["page"]
        )


@dataclass
class LV95Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV95 format."""

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten."""
        return (
            2324800.0 < self.east.coordinate_value < 2847500.0 and 1074000.0 < self.north.coordinate_value < 1302000.0
        )


@dataclass
class LV03Coordinate(Coordinate):
    """Dataclass to represent a coordinate in the LV03 format."""

    def is_valid(self):
        """Reference: https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten#Beispielkoordinaten.

        To account for uncertainties in the conversion of LV03 to LV95, we allow a margin of 2.
        """
        return 324798.0 < self.east.coordinate_value < 847502.0 and 73998.0 < self.north.coordinate_value < 302002.0


class CoordinateExtractor(DataExtractor):
    """Extracts coordinates from a PDF document."""

    feature_name = "coordinate"

    # look for elevation values to the right and/or immediately below the key
    search_right_factor: float = 10
    search_below_factor: float = 3

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " "}

    def get_coordinates_with_x_y_labels(self, lines: list[TextLine], page: int) -> list[Coordinate]:
        """Find coordinates with explicit "X" and "Y" labels from the text lines.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[Coordinate]: all found coordinates
        """
        # In this case, we can allow for some whitespace in between the numbers.
        # In some older borehole profile the OCR may recognize whitespace between two digits.
        pattern_x = regex.compile(r"X[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, flags=regex.IGNORECASE)
        x_matches = CoordinateExtractor._match_text_with_rect(lines, pattern_x)

        pattern_y = regex.compile(r"Y[=:\s]{0,3}" + COORDINATE_ENTRY_REGEX, flags=regex.IGNORECASE)
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
                page=page,
            )
            if coordinates is not None and coordinates.is_valid():
                found_coordinates.append(coordinates)
        return found_coordinates

    def get_coordinates_near_key(self, lines: list[TextLine], page: int) -> list[Coordinate]:
        """Find coordinates from text lines that are close to an explicit "coordinates" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[Coordinate]: all found coordinates
        """
        # find the key that indicates the coordinate information
        coordinate_key_lines = self.find_feature_key(lines)
        extracted_coordinates = []

        for coordinate_key_line in coordinate_key_lines:
            coord_lines = self.get_lines_near_key(lines, coordinate_key_line)
            extracted_coordinates.extend(self.get_coordinates_from_lines(coord_lines, page))

        return extracted_coordinates

    def get_coordinates_from_lines(self, lines: list[TextLine], page: int) -> list[Coordinate]:
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
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[Coordinate]: A list of potential coordinates
        """
        full_regex = regex.compile(
            r"(?:[XY][=:\s]{0,2})?"
            + COORDINATE_ENTRY_REGEX
            + r".{0,4}?[XY]?[=:\s]{0,2}"
            + COORDINATE_ENTRY_REGEX
            + r"\b"
        )
        potential_coordinates = [
            Coordinate.from_values(
                east=float("{}.{}".format("".join(match.groups(default="")[:3]), match.groups(default="")[3])),
                north=float("{}.{}".format("".join(match.groups(default="")[4:-1]), match.groups(default="")[-1])),
                rect=rect,
                page=page,
            )
            for match, rect in CoordinateExtractor._match_text_with_rect(lines, full_regex, self.preprocess)
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

    def extract_coordinates_from_bbox(
        self, page: fitz.Page, page_number: int, bbox: fitz.Rect | None = None
    ) -> Coordinate | None:
        """Extracts the coordinates from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.

        Algorithm description:
            1. search for coordinates with explicit 'X' and 'Y' labels
            2. if that gives no results, search for coordinates close to an explicit "coordinates" label
            3. if that gives no results either, try to detect coordinates in the full text

        Returns:
            Coordinate | None: the extracted coordinates (if any)
        """
        lines = extract_text_lines_from_bbox(page, bbox)

        found_coordinates = (
            self.get_coordinates_with_x_y_labels(lines, page_number)
            or self.get_coordinates_near_key(lines, page_number)
            or self.get_coordinates_from_lines(lines, page_number)
        )

        if len(found_coordinates) > 0:
            return found_coordinates[0]

        logger.info("No coordinates found in this borehole profile.")

    def extract_coordinates(self) -> Coordinate | None:
        """Extracts the coordinates from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.

        Algorithm description:
            1. search for coordinates with explicit 'X' and 'Y' labels
            2. if that gives no results, search for coordinates close to an explicit "coordinates" label
            3. if that gives no results either, try to detect coordinates in the full text

        Returns:
            Coordinate | None: the extracted coordinates (if any)
        """
        for page in self.doc:
            page_number = page.number + 1  # page.number is 0-based

            return self.extract_coordinates_from_bbox(page, page_number)
