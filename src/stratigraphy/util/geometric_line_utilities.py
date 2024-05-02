"""This module contains utility functions to work with geometric lines."""

import logging
from collections import deque
from contextlib import suppress
from itertools import combinations
from math import atan, cos, pi, sin

import numpy as np
import quads
from numpy.typing import ArrayLike
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from stratigraphy.util.dataclasses import IndexedLines, Line, Point

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


def group_lines_by_proximity(lines: list[Line], eps: float = 0.2) -> list[list[Line]]:
    """Group lines that are close to each other using a density clustering (DBSCAN) algorithm.

    Args:
        lines (list[Line]): The lines to group.
        eps (float, optional): Epsilon threshold for the DBSCAN algorithm.. Defaults to 0.2.

    Returns:
        list[list[Line]]: Grouped lines.
    """
    # Calculate the intercepts and slopes of the lines
    features = np.array([(line.slope, line.intercept) for line in lines])

    # use a standardscaler on the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Use DBSCAN to group lines that are close to each other
    db = DBSCAN(eps=eps, min_samples=1).fit(features)

    # Create a list of groups of lines
    groups = []
    for label in set(db.labels_):
        group = [lines[i] for i in range(len(lines)) if db.labels_[i] == label]
        groups.append(group)
    return groups


def merge_parallel_lines_approximately(
    lines: list[Line], tol=8, eps: float = 0.08, angle_threshold: float = 2
) -> list[Line]:
    """Merge parallel lines that are close to each other approximately.

    The algorithm first groups lines together that have a similar slope and intercept using a density clustering
    algorithm. Then the merging problem is solved for each group.

    Given default parameters, the algorithm takes roughly 1 seconds for one pdf page. Without the clustering,
    the algorithm takes roughly 2s for one pdf page.

    Args:
        lines (list[Line]): The lines to merge.
        tol (int, optional): Tolerance to check if lines are close. Defaults to 8.
        eps (float, optional): Epsilon threshold for the DBSCAN algorithm. Defaults to 0.08.
        angle_threshold (float, optional): Acceptable difference between the slopes of two lines. Defaults to 2.

    Returns:
        list[Line]: The merged lines.
    """
    groups = group_lines_by_proximity(lines, eps=eps)
    merged_lines = []
    for group in groups:
        merged_lines.extend(merge_parallel_lines_efficiently(group, tol=tol, angle_threshold=angle_threshold))
    return merged_lines


def merge_parallel_lines_efficiently(lines: list[Line], tol: int, angle_threshold: float) -> list[Line]:
    """Merge parallel lines that are close to each other.

    This merging algorithm first sorts lines by their intercept and slope and then only compares neighbours. This
    reduces the time complexity to O(nlogn). However, 2D lines have no natural ordering, so the algorithm is not
    perfect. However, it manages to quickly merge some lines together.

    The algorithm first looks at neighbours in the slope-intercept space and then uses a sorting using the x and y
    coordinates of the line midpoints. The former is good for long lines, whereas the latter is good for short lines.
    Short lines sometimes have a large difference in their slope and intercept, but they would still fulfill the
    criteria of being parallel and close to each other.

    With default parameters, the algorithm takes roughly 0.5 seconds for one pdf page.

    Note: This function often does not merge lines where line A is short, and line B is long and completely overlaps
    line A. Likely, both sorting objectives do not identify these lines as neighbours. For the purpose of block
    splitting, this does not matter. But for future use of lines this may cause problems. A workaround is to use the
    function merge_parallel_lines which has a higher time complexity but is more complete.

    Args:
        lines (list[Line]): The lines to merge.
        tol (int, optional): Tolerance to check if lines are close.
        angle_threshold (float, optional): Acceptable difference between the slopes of two lines.

    Returns:
        list[Line]: The merged lines.
    """
    # Sort lines by slope and intercept
    lines = merge_parallel_lines_neighbours(
        lines, sorting_function=lambda line: (line.intercept, line.slope), tol=tol, angle_threshold=angle_threshold
    )
    return merge_parallel_lines_neighbours(
        lines,
        sorting_function=lambda line: ((line.start.y + line.end.y) / 2, (line.start.x + line.end.x) / 2),
        tol=tol,
        angle_threshold=angle_threshold,
    )


def merge_parallel_lines_neighbours(
    lines: list[Line], sorting_function: callable, tol: int, angle_threshold: float
) -> list[Line]:
    """Merge parallel lines by comparing neighbours in the sorted list of lines.

    Args:
        lines (list[Line]): The lines to merge.
        sorting_function (callable): The function to sort the lines by.
        tol (int, optional): Tolerance to check if lines are close.
        angle_threshold (float, optional): Acceptable difference between the slopes of two lines.

    Returns:
        list[Line]: The merged lines.
    """
    lines.sort(key=sorting_function)
    merged_lines = []
    current_line = lines[0]
    any_merges = False
    for line in lines[1:]:
        if _are_parallel(current_line, line, angle_threshold=angle_threshold) and _are_close(
            current_line, line, tol=tol
        ):
            merged_line = _merge_lines(current_line, line)
            if merged_line is not None:
                # current_line and line were merged
                current_line = merged_line
                any_merges = True
                continue
        # current_line and line were not merged
        merged_lines.append(current_line)
        current_line = line
    merged_lines.append(current_line)
    if any_merges:
        return merge_parallel_lines_neighbours(
            merged_lines, sorting_function=sorting_function, tol=tol, angle_threshold=angle_threshold
        )
    else:
        return merged_lines


def merge_parallel_lines(lines: list[Line], tol: int, angle_threshold: float) -> list[Line]:
    """Merge parallel lines that are close to each other.

    NOTE: This function can likely be dropped. It is kept here for reference until the line functions
    are finalized.

    This function is most complete but has a high time complexity (O(n^3)). That's why this function is
    typically not called on all lines, but only on a subset.

    Args:
        lines (list[Line]): The lines to merge.
        tol (int, optional): Tolerance to check if lines are close.
        angle_threshold (float, optional): Acceptable difference between the slopes of two lines.

    Returns:
        list[Line]: The merged lines.
    """
    merged_lines = []
    line_queue = deque(lines)
    merged_any = False
    while line_queue:
        line = line_queue.popleft()
        merged = False
        for merge_candidate in merged_lines:
            if _are_parallel(line, merge_candidate, angle_threshold=angle_threshold) and _are_close(
                line, merge_candidate, tol=tol
            ):
                line = _merge_lines(line, merge_candidate)
                if line is None:  # No merge possible
                    merged_lines.append(line)
                    continue
                merged_lines.remove(merge_candidate)
                merged_lines.append(line)
                merged = True
                merged_any = True
        if not merged:
            merged_lines.append(line)
    if merged_any:
        return merge_parallel_lines(merged_lines, tol=tol, angle_threshold=angle_threshold)
    else:
        return merged_lines


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
    indexed_lines = IndexedLines(lines)

    # Create a quadtree
    width = max([line.end.x for line in indexed_lines.hashmap.values()])
    max_end_y = max([line.end.y for line in indexed_lines.hashmap.values()])
    max_start_y = max([line.start.y for line in indexed_lines.hashmap.values()])
    height = max(max_end_y, max_start_y)
    qtree = quads.QuadTree(
        (width / 2, height / 2), width + 1000, height + 1000
    )  # Add some margin to the width and height to allow for slightly negative values
    for line_index, line in indexed_lines.hashmap.items():
        try:
            qtree.insert((line.start.x, line.start.y), data=line_index)
            qtree.insert((line.end.x, line.end.y), data=line_index)
        except ValueError:
            print(f"Could not insert line {line} into quadtree.")

    line_index_stack = list(indexed_lines.hashmap.keys()).copy()
    merged_any = False
    while line_index_stack:
        line_index = line_index_stack.pop(0)
        line = indexed_lines.hashmap[line_index]
        # Get all points in the quadtree that are close to the line
        min_x = min(line.start.x, line.end.x)
        max_x = max(line.start.x, line.end.x)
        min_y = min(line.start.y, line.end.y)
        max_y = max(line.start.y, line.end.y)
        bb = quads.BoundingBox(min_x - tol, min_y - tol, max_x + tol, max_y + tol)

        points = qtree.within_bb(bb)
        merge_candidates_idx = get_merge_candidates_from_points(points, indexed_lines)

        for merge_candidate_idx in merge_candidates_idx:
            if line_index == merge_candidate_idx:  # do not compare line with itself
                continue
            merge_candidate = indexed_lines.hashmap[merge_candidate_idx]

            if _are_parallel(line, merge_candidate, angle_threshold=angle_threshold) and _are_close(
                line, merge_candidate, tol=tol
            ):
                line = _merge_lines(line, merge_candidate)
                indexed_lines.remove(merge_candidate_idx)
                indexed_lines.remove(line_index)
                indexed_lines.add(line)

                # also remove merge_candidate from line_index_stack to avoid uneccessary comparisons
                with suppress(ValueError):  # point might already have been removed from line_index_stack
                    line_index_stack.remove(merge_candidate_idx)

                merged_any = True
    if merged_any:
        return merge_parallel_lines_quadtree(
            list(indexed_lines.hashmap.values()), tol=tol, angle_threshold=angle_threshold
        )
    else:
        return list(indexed_lines.hashmap.values())


def get_merge_candidates_from_points(points: list[quads.Point], lines: IndexedLines) -> set:
    """Obtain the indices of lines that are defined by the points.

    These indices are the merge candidates for the line.

    Args:
        points (list[quads.Point]): The points to check for lines.
        lines (IndexedLines): The lines to check for.

    Returns:
        set: The indices of the lines that are defined by the points.
    """
    merge_idx = set()
    for point in points:
        if point.data in lines.hashmap:
            merge_idx.add(point.data)
    return merge_idx
