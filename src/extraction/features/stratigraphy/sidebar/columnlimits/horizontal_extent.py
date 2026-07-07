"""Module for computing the maximal horizontal extent for a column of depth values based on a table structure."""

from collections.abc import Callable
from dataclasses import dataclass

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line


@dataclass
class HorizontalExtent:
    """Represents how wide a column can be (left or right edge) at a given y-coordiante."""

    extent_from: dict[float, float]
    distance: Callable[[float], float]

    def add_line(self, line: Line) -> "HorizontalExtent":
        """Creates a new extent that is narrower than the current one based on the given vertical line.

        Args:
            line: Line      A vertical line.

        Returns: the new extent
        """
        # TODO: deal with lines that are not perfectly vertical
        line_x = (line.start.x + line.end.x) / 2

        distance = self.distance(line_x)
        if distance < 0:
            # the line is not relevant for this extent (wrong side)
            return self

        y0 = min(line.start.y, line.end.y)

        relevant_points = list(self.extent_from.keys())
        if y0 > min(relevant_points) and distance < self.extent_at(y0):
            relevant_points.append(y0)

        new_extent = {
            y: line_x if distance < self.distance(self.extent_at(y)) and y >= y0 else self.extent_at(y)
            for y in relevant_points
        }
        return HorizontalExtent(new_extent, self.distance)

    def extent_at(self, y: float) -> float | None:
        """Return the extent at the given vertical position.

        Args:
            y: float        The y-coordinate.

        Returns: The corresponding extent, or None if the y-coordinate is above the domain of the extent
        """
        points_above = [key for key in self.extent_from if key <= y]
        if len(points_above):
            return self.extent_from[max(points_above)]
        else:
            return None
