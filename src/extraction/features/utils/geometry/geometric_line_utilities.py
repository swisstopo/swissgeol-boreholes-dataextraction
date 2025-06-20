"""This module contains utility functions to work with geometric lines."""

import logging
import queue
from itertools import combinations
from math import cos, sin

import numpy as np
from numpy.typing import ArrayLike

from .geometry_dataclasses import Line, Point
from .linesquadtree import LinesQuadTree

logger = logging.getLogger(__name__)


def is_point_on_line(line: Line, point: Point, tol=10) -> bool:
    """Check if a point is on a line.

    The check is done by calculating the slope and y-intercept of the line and then checking if the point satisfies
    the equation of the line with some margin tol. Since the lines is only a line segments, the function also checks
    if the point is between the start and end points of the segment. Again, here we allow for some margin, tol / 2 for
    the y-coordinate and tol for the x-coordinate. We assume lines are horizontal, and allow for less margin in the
    y-coordinate to keep merged lines horizontal.

    Args:
        line (Line): a line segment
        point (Point): any point
        tol (int, optional): Tolerance to check if point is on the line. Defaults to 10.

    Returns:
        bool: True if the point is on the line (within the allowed tolerance), False otherwise.
    """
    x_start = np.min([line.start.x, line.end.x])
    x_end = np.max([line.start.x, line.end.x])
    y_start = np.min([line.start.y, line.end.y])
    y_end = np.max([line.start.y, line.end.y])

    # Check if the point satisfies the equation of the line
    return (line.distance_to(point) < tol) and (
        (x_start - tol <= point.x <= x_end + tol) and (y_start - tol <= point.y <= y_end + tol)
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
        # most of those lines are small, and will likelly be deleted at the next step
        logger.debug(
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

    We use an adaptive tolerance to avoid merging very small lines that are relatively far apart compared to
    their size.

    Args:
        line1 (Line): The first line.
        line2 (Line): The second line.
        tol (int): The tolerance to check if the lines are close.

    Returns:
        bool: True if the lines are close, False otherwise.
    """
    min_tol = 2
    adaptive_tol = min(max(min_tol, min(line1.length, line2.length) / 2), tol)
    return (
        is_point_on_line(line1, line2.start, tol=adaptive_tol)
        or is_point_on_line(line1, line2.end, tol=adaptive_tol)
        or is_point_on_line(line2, line1.start, tol=adaptive_tol)
        or is_point_on_line(line2, line1.end, tol=adaptive_tol)
    )


def _are_parallel(line1: Line, line2: Line, angle_threshold: float) -> bool:
    """Check if two lines are parallel.

    Args:
        line1 (Line): The first line.
        line2 (Line): The second line.
        angle_threshold (float, optional): The acceptable difference between the angles of the lines in degrees.

    Returns:
        bool: True if the lines are parallel, False otherwise.
    """
    return (
        np.abs(line1.angle - line2.angle) < angle_threshold
        or 180 - np.abs(line1.angle - line2.angle) < angle_threshold
    )


def _orthogonal_projection_point_on_segment(P: Point, line: Line) -> tuple[np.ndarray, bool]:
    """Projects the point P onto the line (extended) and checks if the projection lies on the segment."""
    P, A, B = (p.as_numpy for p in (P, line.start, line.end))
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    projection = A + t * AB
    is_on_segment = 0 <= t <= 1
    return projection, is_on_segment


def _clamp_projection_to_segment(proj: np.ndarray, line: Line) -> np.ndarray:
    """Clamp a projected point to the segment AB if it lies outside."""
    A, B = (p.as_numpy for p in (line.start, line.end))
    AB = B - A
    t = np.dot(proj - A, AB) / np.dot(AB, AB)
    if t < 0:
        return A
    elif t > 1:
        return B
    else:
        return proj


def _orthogonal_projection_lenght_short_onto_long(short: Line, long: Line) -> float:
    """Computes the length of the orthogonal projection of the short line onto the long one."""
    proj_start, start_on = _orthogonal_projection_point_on_segment(short.start, long)
    proj_end, end_on = _orthogonal_projection_point_on_segment(short.end, long)

    # Clamp projections to segment if they are outside
    if not start_on:
        proj_start = _clamp_projection_to_segment(proj_start, long)
    if not end_on:
        proj_end = _clamp_projection_to_segment(proj_end, long)

    # Check if projections overlap segment
    # If projections are in opposite order along segment, length = 0
    # To check order, project scalar t values for both points:
    long_start, long_end = long.start.as_numpy, long.end.as_numpy
    AB = long_end - long_start
    t_start = np.dot(proj_start - long_start, AB) / np.dot(AB, AB)
    t_end = np.dot(proj_end - long_start, AB) / np.dot(AB, AB)

    if t_start > t_end:
        t_start, t_end = t_end, t_start
        proj_start, proj_end = proj_end, proj_start

    if t_end < 0 or t_start > 1:
        # No overlap with segment
        return 0.0

    # Clamp t values between 0 and 1 to handle floating point imprecision near segment boundaries
    t_start = max(0, t_start)
    t_end = min(1, t_end)

    # Final clipped points
    clipped_start = long_start + t_start * AB
    clipped_end = long_start + t_end * AB

    return np.linalg.norm(clipped_end - clipped_start)


def _following_coefecient(line1: Line, line2: Line):
    long, short = (line1, line2) if line1.length > line2.length else (line2, line1)
    len_proj = _orthogonal_projection_lenght_short_onto_long(short, long)
    overlap_ratio = len_proj / short.length
    return 1 - overlap_ratio


def _are_mergeable(line1: Line, line2: Line, tol: float, angle_threshold: float) -> bool:
    if not _are_parallel(line1, line2, angle_threshold=angle_threshold):
        return False

    perp_tol = tol / 3
    # Adjust distance tolerance based on orientation:
    # When lines follow each other (coef ≈ 1), allow full tolerance (`tol`)
    # When lines are side-by-side (coef ≈ 0), be more strict: only `tol / 3`
    coef = _following_coefecient(line1, line2)
    distance_tolerance = tol * coef + perp_tol * (1 - coef)

    return _are_close(line1, line2, tol=distance_tolerance)


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

    # merging the biggest lines first is more robust
    lines = sorted(lines, key=lambda line: line.length)

    keys_queue = queue.Queue()
    for line in lines:
        line_key = lines_quad_tree.add(line)
        keys_queue.put(line_key)

    while not keys_queue.empty():
        line_key = keys_queue.get()

        if line_key not in lines_quad_tree.hashmap:
            # already seen
            continue
        line = lines_quad_tree.hashmap[line_key]

        neighbours = sorted(
            lines_quad_tree.neighbouring_lines(line_key, tol).items(), key=lambda pair: -pair[1].length
        )  # merging the biggest lines first is more robust
        for neighbour_key, neighbour_line in neighbours:
            if _are_mergeable(line, neighbour_line, tol, angle_threshold):
                new_line = _merge_lines(line, neighbour_line)
                if new_line is not None:
                    lines_quad_tree.remove(neighbour_key)
                    lines_quad_tree.remove(line_key)
                    new_key = lines_quad_tree.add(new_line)
                    keys_queue.put(new_key)
                    break

    return list(lines_quad_tree.hashmap.values())
