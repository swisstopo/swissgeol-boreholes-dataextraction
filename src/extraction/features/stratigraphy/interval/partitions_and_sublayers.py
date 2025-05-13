"""Module for detecting partitions and sublayers in a list of depth intervals."""

from typing import TypeVar

from extraction.features.stratigraphy.interval.interval import Interval, IntervalBlockPair


def number_of_subintervals(interval: Interval, following_intervals: list[Interval]) -> int:
    """Count how many of the following intervals can be considered as sublayers of the given interval."""
    count = 0
    for following_interval in following_intervals:
        if interval.start.value <= following_interval.start.value <= interval.end.value and (
            interval.start.value <= following_interval.end.value <= interval.end.value
        ):
            count += 1
        else:
            break
    return count


def is_partitioned(interval: Interval, following_intervals: list[Interval]) -> bool:
    """Check if the interval is precisely partitioned by some of the following intervals.

    For example, the interval 0m-2m can be partitioned by the three intervals 0m-0.5m, 0.5m-1.5m and 1.5m-2m.
    """
    current_end = interval.start.value
    for following_interval in following_intervals:
        if following_interval.start.value == current_end:
            current_end = following_interval.end.value
            if current_end == interval.end.value:
                return True
            if current_end > interval.end.value:
                return False
        else:
            return False
    return False


T = TypeVar("T", bound=Interval)


def detect_partitions_and_sublayers(intervals: list[T]) -> list[T]:
    """Detect partitions and sublayers in a list of depth intervals.

    Args:
        intervals (list[Interval]): The list of intervals.

    Returns:
        list[Interval]: The list with the relevant attributes set for each interval.
    """
    intervals = intervals.copy()  # don't mutate the original object

    continue_search = True
    while continue_search:
        continue_search = False
        for index, interval in enumerate(intervals):
            if not interval.skip_interval:
                following_intervals = [interval for interval in intervals[index + 1 :] if not interval.skip_interval]
                if is_partitioned(interval, following_intervals):
                    intervals[index].is_parent = True
                    continue_search = True
                    break

    continue_search = True
    while continue_search:
        continue_search = False
        filtered_intervals = [interval for interval in intervals if not interval.skip_interval]
        for index, interval in enumerate(filtered_intervals):
            subinterval_count = number_of_subintervals(interval, filtered_intervals[index + 1 :])
            if subinterval_count > 0:
                for step in range(subinterval_count):
                    filtered_intervals[index + step + 1].is_sublayer = True
                continue_search = True
                break

    return intervals


def detect_partitions_and_sublayers_with_text(pairs: list[IntervalBlockPair]) -> list[IntervalBlockPair]:
    """Detect partitions and sublayers in a list of depth interval, also adjusting text blocks where relevant."""
    pairs = pairs.copy()  # don't mutate the original object
    intervals = [pair.depth_interval for pair in pairs if pair.depth_interval]
    detect_partitions_and_sublayers(intervals)

    processed_pairs = []
    current_block = None
    current_interval = None
    for pair in pairs:
        if pair.depth_interval and pair.depth_interval.skip_interval:
            current_block = current_block.concatenate(pair.block) if current_block else None
        else:
            if current_interval:
                processed_pairs.append(IntervalBlockPair(current_interval, current_block))
                current_block = pair.block
            else:
                current_block = current_block.concatenate(pair.block) if current_block else pair.block
            current_interval = pair.depth_interval

    processed_pairs.append(IntervalBlockPair(current_interval, current_block))
    return processed_pairs
