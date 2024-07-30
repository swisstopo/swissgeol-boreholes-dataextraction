"""This module contains utility functions and classes for TextLine objects."""

from __future__ import annotations

import fitz

from stratigraphy.util.util import read_params, x_overlap_significant_largest

material_description = read_params("matching_params.yml")["material_description"]


class TextWord:
    """Class to represent a word on a specific location on a PDF page.

    A TextWord object consists of a fitz Rectangle object and a string.
    The string is the word that is contained in the rectangle. The rectangles are used
    to represent the location of the word in a PDF document.
    """

    def __init__(self, rect: fitz.Rect, text: str, page: int):
        self.rect = rect
        self.text = text
        self.page_number = page

    def __repr__(self) -> str:
        return f"TextWord({self.rect}, {self.text})"


class TextLine:
    """Class to represent TextLine objects.

    A TextLine object is a collection of DepthInterval objects.
    It is used to represent a line of text in a PDF document.
    """

    def __init__(self, words: list[TextWord]):
        """Initialize the TextLine object.

        Args:
            words (list[TextWord]): The words that make up the line.
            page_number (int): The page number of the line. The first page has idx 1.
        """
        self.rect = fitz.Rect()
        for word in words:
            self.rect.include_rect(word.rect)
        self.words = words
        self.page_number = words[0].page_number

    def is_description(self, material_description):
        """Check if the line is a material description."""
        return any(
            self.text.lower().find(word) > -1 for word in material_description["including_expressions"]
        ) and not any(self.text.lower().find(word) > -1 for word in material_description["excluding_expressions"])

    @property
    def text(self) -> str:
        """Get the text of the line."""
        return " ".join([word.text for word in self.words])

    def __repr__(self) -> str:
        return f"TextLine({self.text}, {self.rect})"

    """
    Check if the current line can be trusted as a stand-alone line, even if it is only a tailing segment of a line that
    was directly extracted from the PDF. This decision is made based on the location (especially x0-coordinates) of the
    lines above and below. If there are enough lines with matching x0-coordinates, then we can assume that this lines
    also belongs to the same "column" in the page layout. This is necessary, because text extraction from PDF sometimes
    extracts text lines too "inclusively", resulting in lines that span across different columns.

    The logic is still not very robust. A more robust solution will be possible once we include line detection as a
    feature in this pipeline as well.
    """

    def is_line_start(self, raw_lines_before: list[TextLine], raw_lines_after: list[TextLine]) -> bool:
        """Check if the current line is the start of a new line."""

        def significant_overlap(line: TextLine) -> bool:
            return x_overlap_significant_largest(line.rect, self.rect, 0.5)

        matching_lines_before = [line for line in raw_lines_before if significant_overlap(line)]
        matching_lines_after = [line for line in raw_lines_after if significant_overlap(line)]

        def count_points(lines: list[TextLine]) -> tuple[int, int]:
            exact_points = 0
            indentation_points = 0
            for other in lines:
                line_height = self.rect.height
                if max(other.rect.y0 - self.rect.y1, self.rect.y0 - other.rect.y1) > 5 * line_height:
                    # too far away vertically
                    return exact_points, indentation_points

                if abs(other.rect.x0 - self.rect.x0) < 0.2 * line_height:
                    exact_points += 1
                elif 0 < other.rect.x0 - self.rect.x0 < 2 * line_height:
                    indentation_points += 1
                else:
                    # other line is more to the left, and significantly more to the right (allowing for indentation)
                    return exact_points, indentation_points
            return exact_points, indentation_points

        # three lines before and three lines after
        exact_points_1, indentation_points_1 = count_points(matching_lines_before[:-4:-1])
        exact_points_2, indentation_points_2 = count_points(matching_lines_after[:3])
        exact_points = exact_points_1 + exact_points_2
        indentation_points = indentation_points_1 + indentation_points_2

        return exact_points >= 3 or (exact_points >= 2 and indentation_points >= 1)

    def to_json(self):
        """Convert the TextLine object to a JSON serializable dictionary."""
        return {
            "text": self.text,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
            "page": self.page_number,
        }
