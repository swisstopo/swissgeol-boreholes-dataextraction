"""Module for computing the area of a PDF page where depth values for a sidebar may be located."""

from dataclasses import dataclass

import pymupdf

from extraction.features.stratigraphy.sidebar.columnlimits.horizontal_extent import HorizontalExtent
from swissgeol_doc_processing.utils.table_detection import TableStructure


@dataclass
class ColumnLimits:
    """Represents the horizontal boundaries (left and right) as well as the lowest position a column may extend to."""

    left_extent: HorizontalExtent
    right_extent: HorizontalExtent
    max_y: float

    def contains(self, point: pymupdf.Point) -> bool:
        left_extent = self.left_extent.extent_at(point.y)
        right_extent = self.right_extent.extent_at(point.y)
        in_extent_domain = left_extent is not None and right_extent is not None

        return in_extent_domain and left_extent < point.x < right_extent and point.y < self.max_y

    @classmethod
    def from_table_structure(
        cls, table_structure: TableStructure, point: pymupdf.Point, font_size: float
    ) -> "ColumnLimits":
        left_extent = HorizontalExtent(
            extent_from={point.y: table_structure.bounding_rect.x0}, distance=lambda x: point.x - x
        )
        right_extent = HorizontalExtent(
            extent_from={point.y: table_structure.bounding_rect.x1}, distance=lambda x: x - point.x
        )
        for vertical_line in table_structure.vertical_lines:
            left_extent = left_extent.add_line(vertical_line)
            right_extent = right_extent.add_line(vertical_line)

        max_y = table_structure.bounding_rect.y1

        horizontal_line_y_values = []
        for horizontal_line in table_structure.horizontal_lines:
            # horizontal line below the reference point
            line_y = (horizontal_line.start.y + horizontal_line.end.y) / 2
            if horizontal_line.start.x < point.x < horizontal_line.end.x:
                top_left = max(min(left_extent.extent_from.keys()), line_y - font_size)
                top_right = max(min(right_extent.extent_from.keys()), line_y - font_size)

                # horizontal line (almost) as wide as the horizontal extent above the line
                wide_enough_left = horizontal_line.start.x < left_extent.extent_at(top_left) + font_size
                wide_enough_right = horizontal_line.end.x > right_extent.extent_at(top_right) - font_size
                if wide_enough_left and wide_enough_right:
                    horizontal_line_y_values.append(line_y)

        last_line_y = None
        for line_y in sorted(horizontal_line_y_values):
            # check only horizontal lines below the reference point, and without another horizontal line closely above
            # (because with lines that are close together, line segment detection is often incomplete).
            if line_y > point.y and (last_line_y is None or line_y - last_line_y > 2 * font_size):
                # Unless the extent remains almost as wide across the horizontal line, we limit the column here
                top_left = max(min(left_extent.extent_from.keys()), line_y - font_size)
                top_right = max(min(right_extent.extent_from.keys()), line_y - font_size)

                top_width = right_extent.extent_at(top_right) - left_extent.extent_at(top_left)
                bottom_width = right_extent.extent_at(line_y + font_size) - left_extent.extent_at(line_y + font_size)
                if abs(top_width - bottom_width) > font_size:
                    max_y = min(max_y, line_y)

            last_line_y = line_y

        return ColumnLimits(left_extent, right_extent, max_y)
