"""Quad tree implementation for efficiently finding lines in a specific area of a page/image."""

import fastquadtree

from swissgeol_doc_processing.geometry.geometry_dataclasses import Line, Point


class LinesQuadTree:
    """Wrapper around the quad tree implementation of the fastquadtree library.

    Enables efficiently finding lines that start or end within a given bounding box.
    """

    def __init__(self, bounds: tuple[float, float, float, float]):
        """Create a LinesQuadTree instance.

        The LinesQuadTree will contain an actual quad tree with the end points of all lines, as well as a hashmap to
        keep track of the lines themselves.

        Args:
            bounds: world bounds for the quad tree sturcture as (min_x, min_y, max_x, max_y).
        """
        self.qtree = fastquadtree.QuadTreeObjects(bounds=bounds, capacity=8)
        self.min_x, self.min_y, self.max_x, self.max_y = bounds

    def remove(self, line: Line):
        """Remove a line from the quad tree.

        If no matching line exists in the quad tree, then the method returns immediately without error.

        Args:
            line (Line): The line to be removed from the quad tree.
        """
        self.qtree.delete_by_object(line)

    def add(self, line: Line):
        """Add a line to the quad tree.

        Args:
            line (Line): the line to be added to the quad tree.
        """
        self._qtree_insert(line.start, line)
        self._qtree_insert(line.end, line)

    def neighbouring_lines(self, line: Line, tol: float) -> list[Line]:
        """Efficiently search for all the lines that have a start or end point close to the given line.

        Args:
            line (Line): The line to search neighbours of.
            tol (float): Tolerance value. Search only for lines with a start or end point that is within this distance
                         from the bounding box formed by the start and end points of the given line.

        Returns:
            list[Line]: The lines that are close to the given line
        """
        min_x = min(line.start.x, line.end.x)
        max_x = max(line.start.x, line.end.x)
        min_y = min(line.start.y, line.end.y)
        max_y = max(line.start.y, line.end.y)
        bb = (min_x - tol, min_y - tol, max_x + tol, max_y + tol)
        points = self.qtree.query(bb)

        neighbouring_lines = []
        for point in points:
            neighbour = point.obj
            if neighbour != line:
                neighbouring_lines.append(neighbour)
        return neighbouring_lines

    def _qtree_insert(self, point: Point, line: Line):
        # Coordinates of merged lines might lie outside the bounds that were initially computed. For the purposes of
        # spacial indexing, we just put these points on the edge of the bounding box.
        x = min(max(self.min_x, point.x), self.max_x - 1)
        y = min(max(self.min_y, point.y), self.max_y - 1)

        self.qtree.insert((x, y), obj=line)
