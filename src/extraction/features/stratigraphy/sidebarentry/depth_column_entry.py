"""Contains a class for depth column entries, which indicate the measured depth of an interface between layers."""

from decimal import Decimal

import pymupdf

from extraction.features.stratigraphy.sidebarentry.sidebar_entry import SidebarEntry


class DepthColumnEntry(SidebarEntry[Decimal]):
    """Represents a depth value extracted from the document.

    DepthColumnEntry are used during the extraction process to hold depth data, which will later be part Intervals
    or Sidebars. Unlike `LayerDepthsEntry`, which is used for visualization after extraction, this class is part
    of the core extraction logic, and is the building block for larger object like Sidebars.
    """

    def __init__(self, value: Decimal, rect: pymupdf.Rect, page_number: int):
        super().__init__(value, rect, page_number)
        self.relative_shift = 0.0

    def __repr__(self) -> str:
        return str(self.value)

    @property
    def has_decimal_point(self) -> bool:
        return self.value.as_tuple().exponent < 0

    @classmethod
    def from_string_value(cls, rect: pymupdf.Rect, string_value: str, page_number: int) -> "DepthColumnEntry":
        """Creates a DepthColumnEntry from a string representation of the value.

        Args:
            rect (pymupdf.Rect): The rectangle that defines where the entry was found on the PDF page.
            string_value (str): A string representation of the value.
            page_number (int): The page number.

        Returns:
            DepthColumnEntry: The depth column entry object.
        """
        clean_value = string_value.replace(",", ".")

        # treat a value of e.g. "60." as Decimal("60.0") instead of Decimal("60")
        if clean_value.endswith("."):
            clean_value += "0"

        return cls(rect=rect, value=abs(Decimal(clean_value)), page_number=page_number)

    @property
    def shifted_rect(self) -> pymupdf.Rect:
        """Returns the bounding box, shifted by its vertical relative_shift.

        Returns:
            pymupdf.Rect: The entry rect, shifted by its relative shift.
        """
        return pymupdf.Rect(
            self.rect.x0, self.rect.y0 + self.relative_shift, self.rect.x1, self.rect.y1 + self.relative_shift
        )
