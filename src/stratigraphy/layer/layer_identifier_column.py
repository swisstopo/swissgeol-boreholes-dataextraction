"""Module for the LayerIdentifierColumn class."""

import re

import fitz
from stratigraphy.lines.line import TextLine


class LayerIdentifierEntry:
    """Class for a layer identifier entry.

    Note: As of now this is very similar to DepthColumnEntry. Refactoring may be desired.
    """

    def __init__(self, rect: fitz.Rect, text: str):
        self.rect = rect
        self.text = text

    def __repr__(self):
        return str(self.text)

    def to_json(self):
        """Convert the layer identifier entry to a JSON serializable format.

        Returns:
            dict: The JSON serializable format of the layer identifier entry.
        """
        return {
            "text": self.text,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
        }


class LayerIdentifierColumn:
    """Class for a layer identifier column."""

    def __init__(self, entries: list[LayerIdentifierEntry]):
        """Initialize the LayerIdentifierColumn object.

        Args:
            entries (list[LayerIdentifierEntry]): The entries corresponding to the layer indices.
        """
        self.entries: list[LayerIdentifierEntry] = entries

    @property
    def max_x0(self) -> float:
        """Get the maximum x0 value of the layer identifier column entries.

        Returns:
            float: The maximum x0 value of the layer identifier column entries.
        """
        return max([rect.x0 for rect in self.rects()])

    @property
    def min_x1(self) -> float:
        """Get the minimum x1 value of the layer identifier column entries.

        Returns:
            float: The minimum x1 value of the layer identifier column entries.
        """
        return min([rect.x1 for rect in self.rects()])

    def rect(self) -> fitz.Rect:
        """Get the rectangle of the layer identifier column.

        Returns:
            fitz.Rect: The rectangle of the layer identifier column.
        """
        x0 = min([rect.x0 for rect in self.rects()])
        x1 = max([rect.x1 for rect in self.rects()])
        y0 = min([rect.y0 for rect in self.rects()])
        y1 = max([rect.y1 for rect in self.rects()])
        return fitz.Rect(x0, y0, x1, y1)

    def rects(self) -> list[fitz.Rect]:
        """Get the rectangles of the layer identifier column entries.

        Returns:
            list[fitz.Rect]: The rectangles of the layer identifier column entries.
        """
        return [entry.rect for entry in self.entries]

    def add_entry(self, entry: LayerIdentifierEntry):
        """Add a new layer identifier column entry to the layer identifier column.

        Args:
            entry (LayerIdentifierEntry): The layer identifier column entry to be added.
        """
        self.entries.append(entry)

    def can_be_appended(self, rect: fitz.Rect) -> bool:
        """Checks if a new layer identifier column entry can be appended to the current layer identifier column.

        The checks are:
        - The width of the new rectangle is greater than the width of the current layer identifier column. Or;
        - The middle of the new rectangle is within the horizontal boundaries of the current layer identifier column.
        - The new rectangle intersects with the minimal horizontal boundaries of the current layer identifier column.


        Args:
            rect (fitz.Rect): Rect of the layer identifier column entry to be appended.

        Returns:
            bool: True if the new layer identifier column entry can be appended, False otherwise.
        """
        new_middle = (rect.x0 + rect.x1) / 2
        if (self.rect().width < rect.width or self.rect().x0 < new_middle < self.rect().x1) and (
            rect.x0 <= self.min_x1 and self.max_x0 <= rect.x1
        ):
            return True
        return False

    def strictly_contains(self, other: "LayerIdentifierColumn") -> bool:
        """Check if the layer identifier column strictly contains another layer identifier column.

        Args:
            other (LayerIdentifierColumn): The other layer identifier column to check if it is strictly contained.

        Returns:
            bool: True if the layer identifier column strictly contains the other layer identifier column, False
            otherwise.
        """
        return len(other.entries) < len(self.entries) and all(
            other_entry in self.entries for other_entry in other.entries
        )

    def is_contained(self, rect: fitz.Rect) -> bool:
        """Check if the layer identifier column is contained in another rectangle.

        Args:
            rect (fitz.Rect): The rectangle to check if it contains the layer identifier column.

        Returns:
            bool: True if the layer identifier column is contained in the rectangle, False otherwise.
        """
        return (
            rect.x0 <= self.rect().x0
            and self.rect().x1 <= rect.x1
            and rect.y0 <= self.rect().y0
            and self.rect().y1 <= rect.y1
        )

    def to_json(self):
        """Convert the layer identifier column to a JSON serializable format.

        Returns:
            dict: The JSON serializable format of the layer identifier column.
        """
        rect = self.rect()
        return {
            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
            "entries": [entry.to_json() for entry in self.entries],
            "type": "LayerIdentifierColumn",
        }

    @classmethod
    def from_json(cls, data: dict) -> "LayerIdentifierColumn":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary containing 'entries' list with 'rect' and 'text' fields.

        Raises:
             ValueError: If the input dictionary is missing required fields or has invalid data.

        Returns:
            LayerIdentifierColumn: The layer identifier column object.
        """
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("Invalid input: data must be a dictionary with 'entries' field")

        return LayerIdentifierColumn(
            entries=[
                LayerIdentifierEntry(rect=fitz.Rect(entry["rect"]), text=entry["text"]) for entry in data["entries"]
            ]
        )


def find_layer_identifier_column_entries(lines: list[TextLine]) -> list[LayerIdentifierEntry]:
    r"""Find the layer identifier column entries.

    Regex explanation:
    - \b is a word boundary. This ensures that the match must start at the beginning of a word.
    - [\da-z]+ matches one or more (+) alphanumeric characters (\d for digits and a-z for lowercase letters).
    - \) matches a closing parenthesis. The backslash is necessary because parentheses are special characters
      in regular expressions, so we need to escape it to match a literal parenthesis.
    This regular expression will match strings like "1)", "2)", "a)", "b)", "1a4)", "6de)", etc.

    Args:
        lines (list[TextLine]): The lines to search for layer identifier columns.

    Returns:
        list[LayerIdentifierEntry]: The layer identifier column entries.
    """
    entries = []
    for line in sorted(lines, key=lambda line: line.rect.y0):
        if len(line.words) > 0:
            # Only match in the first word of every line, to avoid e.g. matching with "cm)" in a material description
            # containing an expression like "(diameter max 6 cm)".
            first_word = line.words[0]
            regex = re.compile(r"\b[\da-z-]+\)")
            match = regex.match(first_word.text)
            if match and len(first_word.text) < 7:
                entries.append(LayerIdentifierEntry(first_word.rect, first_word.text))
    return entries


def find_layer_identifier_column(entries: list[LayerIdentifierEntry]) -> list[LayerIdentifierColumn]:
    """Find the layer identifier column given the index column entries.

    Note: Similar to find_depth_columns.find_depth_columns. Refactoring may be desired.

    Args:
        entries (list[LayerIdentifierEntry]): The layer identifier column entries.

    Returns:
        list[LayerIdentifierColumn]: The found layer identifier columns.
    """
    layer_identifier_columns = [LayerIdentifierColumn([entries[0]])]
    for entry in entries[1:]:
        has_match = False
        for column in layer_identifier_columns:
            if column.can_be_appended(entry.rect):
                column.add_entry(entry)
                has_match = True
        if not has_match:
            layer_identifier_columns.append(LayerIdentifierColumn([entry]))

        # only keep columns whose entries are not fully contained in a different column
        layer_identifier_columns = [
            column
            for column in layer_identifier_columns
            if all(not other.strictly_contains(column) for other in layer_identifier_columns)
        ]
        # check if the column rect is a subset of another column rect. If so, merge the entries and sort them by y0.
        for column in layer_identifier_columns:
            for other in layer_identifier_columns:
                if column != other and column.is_contained(other.rect()):
                    for entry in other.entries:
                        if entry not in column.entries:
                            column.entries.append(entry)
                    column.entries.sort(key=lambda entry: entry.rect.y0)
                    layer_identifier_columns.remove(other)
                    break
    layer_identifier_columns = [column for column in layer_identifier_columns if len(column.entries) > 2]
    return layer_identifier_columns
