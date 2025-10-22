"""This module contains dataclasses for (depth) intervals."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from extraction.features.utils.text.textblock import TextBlock

from ..base.sidebar_entry import DepthColumnEntry


class Interval:
    """Class for (depth) intervals.

    This class defines a generic interface for any depth interval, either derived from vertical positions on the page
    (e.g., column-aligned entries) or parsed inline from text (e.g., "1.00 - 2.30m").

    Unlike `LayerDepths`, which is used in visual layout and representation, `Interval` is part of the data
    extraction pipeline.
    """

    def __init__(self, start: DepthColumnEntry | None, end: DepthColumnEntry | None):
        super().__init__()
        self.start = start
        self.end = end
        self.is_parent = False
        self.is_sublayer = False

    def __repr__(self):
        return f"({self.start}, {self.end})"

    @property
    def skip_interval(self) -> bool:
        return self.is_parent or self.is_sublayer


class AAboveBInterval(Interval):
    """Class for depth intervals where the upper depth is located above the lower depth on the page."""

    pass


class AToBInterval(Interval):
    """Class for intervals that are defined in a single line like "1.00 - 2.30m"."""

    @property
    def rect(self) -> pymupdf.Rect:
        """Get the rectangle surrounding the interval."""
        return pymupdf.Rect(self.start.rect).include_rect(self.end.rect)


class SpulprobeInterval(Interval):
    """Class for depth intervals where the delimitations are Spulprobe tags."""

    pass


@dataclass
class IntervalBlockPair:
    """Represent the data for a single layer in the borehole profile.

    This consist of a material description (represented as a text block) and a depth interval (if available).
    """

    depth_interval: Interval | None
    block: TextBlock


@dataclass
class IntervalZone:
    """Zone on the page used to match description lines to the correct interval.

    It has more context than a regular interval, as it also considers the interval after to infer the end zone for
    some sidebar types. For example, an AtoB interval have its start and end rect on the same place, where its
    corresponding IntervalZone starts at the interval rect, and ends with the next AtoB interval.
    """

    start: pymupdf.Rect | None
    end: pymupdf.Rect | None
    related_interval: Interval
