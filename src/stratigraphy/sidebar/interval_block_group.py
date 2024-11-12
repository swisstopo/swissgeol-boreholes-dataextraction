"""Module that contains a helper class for associating depth intervals and text blocks."""

from dataclasses import dataclass

from stratigraphy.text.textblock import TextBlock
from stratigraphy.util.interval import Interval


@dataclass
class IntervalBlockGroup:
    """Helper class to represent a group of depth intervals and an associated group of text blocks.

    The class is used to simplify the code for obtaining an appropriate one-to-one correspondence between depth
    intervals and material descriptions.
    """

    depth_intervals: list[Interval]
    blocks: list[TextBlock]
