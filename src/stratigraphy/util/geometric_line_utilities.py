"""This module contains utility functions to work with geometric lines."""

import logging
import queue
from itertools import combinations
from math import atan, cos, pi, sin

import numpy as np
from numpy.typing import ArrayLike

from stratigraphy.util.dataclasses import Line, Point
from stratigraphy.util.linesquadtree import LinesQuadTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_vertical_lines(lines: list[Line], threshold: float = 0.1) -> ArrayLike:
    """Given a list of lines, remove the lines that are close to vertical.

    The algorithm will drop every line whose absolute slope is larger than 1 / threshold.

    Args:
        lines (ArrayLike): The lines to remove the vertical lines from.
        threshold (float, optional): The threshold to determine if a line is vertical. The larger the threshold,
                                     the fewer lines are considered. Defaults to 0.1.

    Returns:
        ArrayLike: The lines with the vertical lines removed.
    """
    return [line for line in lines if np.abs(line.slope) < 1 / threshold]


def is_point_on_line(line: Line, point: Point, tol=10) -> ArrayLike:
    """Check if a point is on a line.

    The check is done by calculating the slope and y-intercept of the line and then checking if the point satisfies
    the equation of the line with some margin tol. Since the lines are only line segments,
    the function also checks if the respecting end points of the line segments overlap. Again, here we
    allow for some margin, tol / 2 for the y-coordinate and tol for the x-coordinate. We assume lines are horizontal,
    and allow for less margin in the y-coordinate to keep merged lines horizontal.

    Args:
        line (Line): Line as detected by LSD:
        point (Point): any point
        tol (int, optional): Tolerance to check if point is on line. Defaults to 10.

    Returns:
        ArrayLike: True if the point is on the line, False otherwise.
    """
    x_start = np.min([line.start.x, line.end.x])
    x_end = np.max([line.start.x, line.end.x])
    y_start = np.min([line.start.y, line.end.y])
    y_end = np.max([line.start.y, line.end.y])

    # Check if the point satisfies the equation of the line
    return (line.distance_to(point) < tol) and (
        (x_start - tol <= point.x <= x_end + tol) and (y_start - tol / 2 <= point.y <= y_end + tol / 2)
    )


def _odr_regression(x: ArrayLike, y: ArrayLike, weights: ArrayLike = None) -> tuple:
    """Perform orthogonal distance regression (a.k.a. Deming regression) on the given data.

    This algorithm minimises the quadratic distance from the given points to the resulting line, where each point
    can be given a specific weight. If no weights are specified, then all points are weighted equally.

    The implementation follow the paper "A tutorial on the total least squares method for fitting a straight line and
    a plane" (https://www.researchgate.net/publication/272179120).

    Note: If the problem is ill-defined (i.e. denominator == nominator == 0),
    then the function will return phi=np.nan and r=np.nan.

    Args:
        x (ArrayLike): The x-coordinates of the data.
        y (ArrayLike): The y-coordinates of the data.
        weights (ArrayLike, optional): The weight for each data point. Defaults to None.

    Returns:
        tuple: (phi, r), the best fit values for the line equation in normal form.
    """
    if weights is None:
        weights = np.ones((len(x),))

    x_mean = np.mean(np.dot(weights, x)) / np.sum(weights)
    y_mean = np.mean(np.dot(weights, y)) / np.sum(weights)
    nominator = -2 * np.sum(np.dot(weights, (x - x_mean) * (y - y_mean)))
    denominator = np.sum(np.dot(weights, (y - y_mean) ** 2 - (x - x_mean) ** 2))
    if nominator == 0 and denominator == 0:
        logger.warning(
            "The line merging problem is ill defined as both nominator and denominator for arctan are 0. "
            "We return phi=np.nan and r=np.nan."
        )
        return np.nan, np.nan
    phi = 0.5 * np.arctan2(nominator, denominator)  # np.arctan2 can deal with np.inf due to zero division.
    r = x_mean * cos(phi) + y_mean * sin(phi)

    if r < 0:
        r = -r
        phi = phi + np.pi

    return phi, r


def _merge_lines(line1: Line, line2: Line) -> Line | None:
    """Merge two lines into one.

    The algorithm performs odr regression on the points of the two lines to find the best fit line.
    Then, it calculates the orthogonal projection of the four points onto the best fit line and takes the two points
    that are the furthest apart. These two points are then used to create the merged line.

    Note: a few cases (e.g. the lines are sides of a perfect square) the solution is not well-defined. In such a case,
    the method returns None.

    Args:
        line1 (Line): First line to merge.
        line2 (Line): Second line to merge.

    Returns:
        Line | None: The merged line.
    """
    x = np.array([line1.start.x, line1.end.x, line2.start.x, line2.end.x])
    y = np.array([line1.start.y, line1.end.y, line2.start.y, line2.end.y])
    # For the orthogonal distance regression, we weigh each point proportionally to the length of the line. The
    # intuition behind this, is that we want the resulting merged line to be equal, regardless of whether we are
    # merging a line AB with a line CD, or whether we are merging three lines AB, CE and ED, where E is the midpoint
    # of CD. The current choice of weights only achieves this requirement if the lines are exactly parallel. Still,
    # we use these weights also for non-parallel input lines, as it is still a good enough approximation for our
    # purposes, and it keeps the mathematics reasonable simple.
    weights = np.array([line1.length, line1.length, line2.length, line2.length])
    phi, r = _odr_regression(x, y, weights)
    if np.isnan(phi) or np.isnan(r):
        return None

    projected_points = []
    for point in [line1.start, line1.end, line2.start, line2.end]:
        projection = _get_orthogonal_projection_to_line(point, phi, r)
        projected_points.append(projection)
    # Take the two points such that the distance between them is the largest.
    # Since there are only 4 points involved, we can just calculate the distance between all points.
    point_combinations = list(combinations(projected_points, 2))
    distances = []
    for interval in point_combinations:
        distances.append(_calculate_squared_distance_between_two_points(interval[0], interval[1]))

    point1, point2 = point_combinations[np.argmax(distances)]
    return Line(point1, point2)


def _calculate_squared_distance_between_two_points(point1: Point, point2: Point) -> float:
    return (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2


def _get_orthogonal_projection_to_line(point: Point, phi: float, r: float) -> Point:
    """Calculate orthogonal projection of a point onto a line in 2D space.

    Calculates the orhtogonal projection of a point onto a line in 2D space using the normal form of the line equation.
    Formula derived from:
    https://www.researchgate.net/publication/272179120
    _A_tutorial_on_the_total_least_squares_method_for_fitting_a_straight_line_and_a_plane

    Args:
        point (Point): The point to project onto the line.
        phi (float): The angle phi of the normal form of the line equation.
        r (float): The distance r of the normal form of the line equation.

    Returns:
        Point: The projected point onto the line.
    """
    # ri is the distance to the line that is parallel to the line defined by phi and r
    # and that goes through the poin (point.y, point.y).
    ri = point.x * cos(phi) + point.y * sin(phi)
    d = ri - r  # d is the distance between point and the line.
    # We now need to move towards the line by d in the direction defined by phi.
    x = point.x - d * cos(phi)
    y = point.y - d * sin(phi)
    return Point(x, y)


def _are_close(line1: Line, line2: Line, tol: int) -> bool:
    """Check if two lines are close to each other.

    Args:
        line1 (Line): The first line.
        line2 (Line): The second line.
        tol (int, optional): The tolerance to check if the lines are close.

    Returns:
        bool: True if the lines are close, False otherwise.
    """
    return is_point_on_line(line1, line2.start, tol=tol) or is_point_on_line(line1, line2.end, tol=tol)


def _are_parallel(line1: Line, line2: Line, angle_threshold: float) -> bool:
    """Check if two lines are parallel.

    Args:
        line1 (Line): The first line.
        line2 (Line): The second line.
        angle_threshold (float, optional): The acceptable difference between the slopes of the lines.

    Returns:
        bool: True if the lines are parallel, False otherwise.
    """
    return np.abs(atan(line1.slope) - atan(line2.slope)) < angle_threshold * pi / 180


def merge_parallel_lines_quadtree(lines: list[Line], tol: int, angle_threshold: float) -> list[Line]:
    """Merge parallel lines that are close to each other.

    Uses a quadtree to quickly find lines that are close to each other. The algorithm is more efficient than the
    naive approach.

    Args:
        lines (list[Line]): The lines to merge.
        tol (int, optional): Tolerance to check if lines are close.
        angle_threshold (float, optional): Acceptable difference between the slopes of two lines.

    Returns:
        list[Line]: The merged lines.
    """
    # Create a quadtree
    width = max([line.end.x for line in lines])
    max_end_y = max([line.end.y for line in lines])
    max_start_y = max([line.start.y for line in lines])
    height = max(max_end_y, max_start_y)
    lines_quad_tree = LinesQuadTree(width, height)

    keys_queue = queue.Queue()
    for line in lines:
        line_key = lines_quad_tree.add(line)
        keys_queue.put(line_key)

    while not keys_queue.empty():
        line_key = keys_queue.get()

        if line_key in lines_quad_tree.hashmap:
            line = lines_quad_tree.hashmap[line_key]

            for neighbour_key, neighbour_line in lines_quad_tree.neighbouring_lines(line_key, tol).items():
                if _are_parallel(line, neighbour_line, angle_threshold=angle_threshold) and _are_close(
                    line, neighbour_line, tol=tol
                ):
                    new_line = _merge_lines(line, neighbour_line)
                    if new_line is not None:
                        lines_quad_tree.remove(neighbour_key)
                        lines_quad_tree.remove(line_key)
                        new_key = lines_quad_tree.add(new_line)
                        keys_queue.put(new_key)
                        break

    return list(lines_quad_tree.hashmap.values())
