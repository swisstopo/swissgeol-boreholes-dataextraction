"""Dataclasses for stratigraphy module."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """Class to represent a point in 2D space."""

    x: float
    y: float

    @property
    def tuple(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def as_numpy(self) -> np.ndarray:
        return np.array(self.tuple)

    def distance_to(self, point: Point) -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


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

        self.length = self.start.distance_to(self.end)

    def distance_to(self, point: Point) -> float:
        """Calculate the distance of a point to the (unbounded extension of the) line.

        Taken from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

        Args:
            point (Point): The point to calculate the distance to.

        Returns:
            float: The distance of the point to the line.
        """
        return np.abs(
            (self.end.x - self.start.x) * (self.start.y - point.y)
            - (self.start.x - point.x) * (self.end.y - self.start.y)
        ) / np.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)

    @property
    def slope(self) -> float:
        """Calculate the slope of the line."""
        return (self.end.y - self.start.y) / (self.end.x - self.start.x) if self.end.x - self.start.x != 0 else np.inf

    @property
    def angle(self) -> float:
        """Angle of the line with the x-axis in degrees, ranging from -90 (exclusive) to +90 (inclusive)."""
        if self.start.x == self.end.x:
            return 90
        else:
            return math.atan(self.slope) / math.pi * 180

    @property
    def intercept(self) -> float:
        """Calculate the y-intercept of the line."""
        return self.start.y - self.slope * self.start.x


@dataclass
class BoundingBox:
    """A single bounding box, JSON serializable."""

    rect: pymupdf.Rect

    def to_json(self) -> list[int]:
        """Converts the object to a dictionary.

        Returns:
            list[int]: The object as a list.
        """
        return [
            self.rect.x0,
            self.rect.y0,
            self.rect.x1,
            self.rect.y1,
        ]

    @classmethod
    def from_json(cls, data) -> BoundingBox:
        return cls(rect=pymupdf.Rect(data))
