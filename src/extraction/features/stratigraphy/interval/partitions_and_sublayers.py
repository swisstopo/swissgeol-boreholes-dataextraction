"""Module for detecting partitions and sublayers in a list of depth intervals."""

from typing import TypeVar

from extraction.features.stratigraphy.interval.interval import Interval, IntervalBlockPair


def number_of_subintervals(interval: Interval, following_intervals: list[Interval]) -> int:
    """Counts how many of the following intervals are completely contained within a given interval.

    Args:
        interval (Interval): The main interval to compare against.
        following_intervals (list[Interval]): A list of subsequent intervals to check.

    Returns:
        int: The number of contiguous intervals in `following_intervals` that are fully contained in `interval`.
    """
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
    """Determines whether a sequence of following intervals forms a complete partition of the given interval.

    Args:
        interval (Interval): The interval to test for partitioning.
        following_intervals (list[Interval]): A list of subsequent intervals.

    Returns:
        bool: True if the subsequent intervals exactly partition the input interval, False otherwise.
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


def annotate_intervals(intervals: list[T]) -> None:
    """Mutate each Interval object by setting the flags is parent or is_sublayer.

    This function marks is_parent any interval that fully partitions one or more later intervals.
    It also marks is_sublayer any interval that is contained in another but is not part of it's partition.

    Args:
        intervals (list[T]): the list of interval to analyse and mutate the flags.
    """
    # 1) Mark parents
    while True:
        for i, current_interval in enumerate(intervals):
            if current_interval.skip_interval:
                continue
            # all later, non-skipped intervals
            following_intervals = [interval for interval in intervals[i + 1 :] if not interval.skip_interval]
            if is_partitioned(current_interval, following_intervals):
                current_interval.is_parent = True
                break
        else:
            # if the for-loop completes without breaking, this else condition is evaluated, and breaks the while-loop
            break

    # 2) Mark sublayers
    while True:
        # work only on the non-skipped list
        valid_intervals = [interval for interval in intervals if not interval.skip_interval]
        for i, current_interval in enumerate(valid_intervals):
            subinterval_count = number_of_subintervals(current_interval, valid_intervals[i + 1 :])
            if subinterval_count:
                for child_interval in valid_intervals[i + 1 : i + 1 + subinterval_count]:
                    child_interval.is_sublayer = True
                break
        else:
            # if the for-loop completes without breaking, this else condition is evaluated, and breaks the while-loop
            break


def aggregate_non_skipped_intervals(
    pairs: list[IntervalBlockPair],
) -> list[IntervalBlockPair]:
    """Build a new list of IntervalBlockPair where skipped intervals blocks.

    are merged into the last non-skipped intervals block.
    """
    processed_pairs: list[IntervalBlockPair] = []
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


def get_optimal_intervals_with_text(
    pairs: list[IntervalBlockPair],
) -> list[IntervalBlockPair]:
    """Annotate all depth_intervals in-place for parent/sublayer and merge valid the intervals and text blocks.

    Args:
        pairs (list[IntervalBlockPair]): list of intervals and text blocks.

    Return:
        list[IntervalBlockPair]: the cleaned list, with optimal partitions filtered and text block conserved.
    """
    # 1) annotate the intervals you have
    annotate_intervals([p.depth_interval for p in pairs if p.depth_interval])

    # 2) rebuild & merge text blocks
    return aggregate_non_skipped_intervals(pairs)
