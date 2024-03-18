""" Dataclasses for stratigraphy module. """

from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    """Class to represent a point in 2D space."""

    x: float
    y: float

    @property
    def tuple(self) -> (float, float):
        return self.x, self.y


@dataclass
class Line:
    """Class to represent a line in 2D space."""

    start: Point
    end: Point

    def __post_init__(self):
        if self.start.x > self.end.x:
            end = self.start
            self.start = self.end
            self.end = end

        self.slope = self.slope()
        self.intercept = self.intercept()

    def distance_to(self, point: Point) -> float:
        # Calculate the distance of the point to the line:
        #  Taken from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        return np.abs(
            (self.end.x - self.start.x) * (self.start.y - point.y)
            - (self.start.x - point.x) * (self.end.y - self.start.y)
        ) / np.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)

    def slope(self) -> float:
        return (self.end.y - self.start.y) / (self.end.x - self.start.x) if self.end.x - self.start.x != 0 else np.inf

    def intercept(self) -> float:
        return self.start.y - self.slope * self.start.x
