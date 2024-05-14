"""Quad tree implementation for efficiently finding lines in a specific area of a page/image."""

import uuid

import quads

from stratigraphy.util.dataclasses import Line, Point


class LinesQuadTree:
    """Wrapper around the quad tree implementation of the quads library.

    Enables efficiently finding lines that start or end within a given bounding box.
    """

    def __init__(self, width: float, height: float):
        self.qtree = quads.QuadTree(
            (width / 2, height / 2), width + 1000, height + 1000
        )  # Add some margin to the width and height to allow for slightly negative values
        self.hashmap = {}

    def remove(self, line_key: str):
        if line_key in self.hashmap:
            line = self.hashmap[line_key]
            self._qtree_delete((line.start.x, line.start.y), line_key)
            self._qtree_delete((line.end.x, line.end.y), line_key)
            del self.hashmap[line_key]

    def add(self, line: Line) -> str:
        line_key = uuid.uuid4().hex
        self.hashmap[line_key] = line

        self._qtree_insert((line.start.x, line.start.y), line_key)
        self._qtree_insert((line.end.x, line.end.y), line_key)
        return line_key

    def neighbouring_lines(self, line_key: str, tol: float) -> list[(str, Line)]:
        """Efficiently search for all the lines that have a start or end point close to the given line.

        Args:
            line_key (keys): The key of the line to search neighbours of.
            tol (float): Tolerance value. Search only for lines with a start or end point that is within this distance
                         from the bounding box formed by the start and end points of the given line.

        Returns:
            list[(str, Line)]: The lines that are close to the given line, returned as a tuples (line_key, line).
        """
        if line_key not in self.hashmap:
            return []

        line = self.hashmap[line_key]
        min_x = min(line.start.x, line.end.x)
        max_x = max(line.start.x, line.end.x)
        min_y = min(line.start.y, line.end.y)
        max_y = max(line.start.y, line.end.y)
        bb = quads.BoundingBox(min_x - tol, min_y - tol, max_x + tol, max_y + tol)
        points = self.qtree.within_bb(bb)

        neighbouring_lines = []
        for point in points:
            for neighbour_key in point.data:
                if neighbour_key != line_key and neighbour_key in self.hashmap:
                    neighbouring_lines.append((neighbour_key, self.hashmap[neighbour_key]))
        return neighbouring_lines

    def _qtree_insert(self, point: Point | tuple, line_key: str):
        qtree_point = self.qtree.find(point)
        if qtree_point:
            qtree_point.data.add(line_key)
        else:
            self.qtree.insert(point, data={line_key})

    def _qtree_delete(self, point: Point | tuple, line_key: str):
        qtree_point = self.qtree.find(point)
        if qtree_point:
            qtree_point.data.remove(line_key)
