"""Dataclasses for stratigraphy module."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

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
        if self.length == 0:
            return self.start.distance_to(point)
        else:
            return (
                np.abs(
                    (self.end.x - self.start.x) * (self.start.y - point.y)
                    - (self.start.x - point.x) * (self.end.y - self.start.y)
                )
                / self.length
            )

    def distance_to_segment(self, point: Point) -> float:
        """Calculate the distance of a point to the line segment (bounded by endpoints).

        Uses vector projection with parametric line representation to find the closest
        point on the segment, then calculates Euclidean distance.

        Args:
            point (Point): The point to calculate the distance to.

        Returns:
            float: The distance of the point to the line segment.
        """
        # Vector from line start to point
        px, py = point.x - self.start.x, point.y - self.start.y
        # Line direction vector
        lx, ly = self.end.x - self.start.x, self.end.y - self.start.y

        line_length_sq = lx * lx + ly * ly
        if line_length_sq == 0:
            # Degenerate case: line is actually a point
            return ((point.x - self.start.x) ** 2 + (point.y - self.start.y) ** 2) ** 0.5

        # Project point onto line using dot product
        t = (px * lx + py * ly) / line_length_sq

        # Clamp t to [0, 1] to stay within segment bounds
        t = max(0, min(1, t))

        # Find closest point on segment
        proj_x = self.start.x + t * lx
        proj_y = self.start.y + t * ly

        # Calculate Euclidean distance
        return ((point.x - proj_x) ** 2 + (point.y - proj_y) ** 2) ** 0.5

    def point_near_segment(self, point: Point, threshold: float) -> bool:
        """Check if a point is within a threshold distance of the line segment.

        Args:
            point (Point): The point to check
            threshold (float): Distance threshold for proximity

        Returns:
            bool: True if the point is within threshold distance of the segment
        """
        return self.distance_to_segment(point) <= threshold

    def intersects_with(self, other: Line) -> bool:
        """Check if this line segment intersects with another line segment.

        Uses line-line intersection calculation with determinants.
        Reference: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        Args:
            other (Line): The other line segment to check intersection with

        Returns:
            bool: True if line segments intersect, False otherwise
        """
        # Get line endpoints
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x3, y3 = other.start.x, other.start.y
        x4, y4 = other.end.x, other.end.y

        # Calculate line intersection using determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Lines are parallel if denominator is 0
        if abs(denom) < 1e-10:
            return False

        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Lines intersect if both parameters are between 0 and 1
        return 0 <= t <= 1 and 0 <= u <= 1

    @property
    def slope(self) -> float:
        """Calculate the slope of the line."""
        return (self.end.y - self.start.y) / (self.end.x - self.start.x) if self.end.x - self.start.x != 0 else np.inf

    def is_horizontal(self, horizontal_slope_tolerance) -> bool:
        """Checks if a line is horizontal."""
        return abs(self.slope) <= horizontal_slope_tolerance

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


@dataclass
class RectWithPage:
    """Dataclass to store a rectangle and the page number it appears on."""

    rect: pymupdf.Rect | None
    page_number: int


class SupportsRectWithPage(Protocol):
    """Protocol to ensure that a class has a rect_with_page attribute."""

    rect_with_page: RectWithPage


class RectWithPageMixin:
    """Mixin class to facilitate the access to the rect and page_number of a SupportsRectWithPage object."""

    @property
    def rect(self: SupportsRectWithPage):
        return self.rect_with_page.rect

    @property
    def page_number(self: SupportsRectWithPage):
        return self.rect_with_page.page_number
