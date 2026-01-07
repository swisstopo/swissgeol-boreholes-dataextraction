"""Quad tree implementation for efficiently finding lines in a specific area of a page/image."""

import uuid

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
        self.hashmap = {}

    def remove(self, line_key: str):
        """Remove a line from the quad tree.

        If no matching line exists in the quad tree, then the method returns immediately without error.

        Args:
            line_key (str): The key of the line to be removed from the quad tree.
        """
        if line_key in self.hashmap:
            line = self.hashmap[line_key]
            self._qtree_delete(line.start, line_key)
            self._qtree_delete(line.end, line_key)
            del self.hashmap[line_key]

    def add(self, line: Line) -> str:
        """Add a line to the quad tree.

        Args:
            line (Line): the line to be added to the quad tree.

        Returns:
            str: the UUID key that is automatically generated for the new line.

        """
        line_key = uuid.uuid4().hex
        self.hashmap[line_key] = line

        self._qtree_insert(line.start, line_key)
        self._qtree_insert(line.end, line_key)
        return line_key

    def neighbouring_lines(self, line_key: str, tol: float) -> dict[str, Line]:
        """Efficiently search for all the lines that have a start or end point close to the given line.

        Args:
            line_key (keys): The key of the line to search neighbours of.
            tol (float): Tolerance value. Search only for lines with a start or end point that is within this distance
                         from the bounding box formed by the start and end points of the given line.

        Returns:
            dict[str, Line]: The lines that are close to the given line, returned as a dict of (line_key, line) pairs.
        """
        if line_key not in self.hashmap:
            return dict()

        line = self.hashmap[line_key]
        min_x = min(line.start.x, line.end.x)
        max_x = max(line.start.x, line.end.x)
        min_y = min(line.start.y, line.end.y)
        max_y = max(line.start.y, line.end.y)
        bb = (min_x - tol, min_y - tol, max_x + tol, max_y + tol)
        points = self.qtree.query(bb)

        neighbouring_lines = {}
        for point in points:
            neighbour_key = point.obj
            if neighbour_key != line_key and neighbour_key in self.hashmap:
                neighbouring_lines[neighbour_key] = self.hashmap[neighbour_key]
        return neighbouring_lines

    def _qtree_insert(self, point: Point, line_key: str):
        # Coordinates of merged lines might lie outside the bounds that were initially computed. For the purposes of
        # spacial indexing, we just put these points on the edge of the bounding box.
        x = point.x
        if x > self.max_x - 1:
            x = self.max_x - 1
        if x < self.min_x:
            x = self.min_x

        y = point.y
        if y >= self.max_y - 1:
            y = self.max_y - 1
        if y < self.min_y:
            y = self.min_y

        self.qtree.insert((x, y), obj=line_key)

    def _qtree_delete(self, point: Point, line_key: str):
        self.qtree.delete_by_object(line_key)
