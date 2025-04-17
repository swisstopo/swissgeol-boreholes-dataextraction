"""Module that contains a helper class for associating depth intervals and text blocks."""

from dataclasses import dataclass

from borehole_extraction.extraction.stratigraphy.depth.interval import DepthInterval
from borehole_extraction.extraction.util_extraction.text.textblock import TextBlock


@dataclass
class IntervalBlockGroup:
    """Helper class to represent a group of depth intervals and an associated group of text blocks.

    The class is used to simplify the code for obtaining an appropriate one-to-one correspondence between depth
    intervals and material descriptions.
    """

    depth_intervals: list[DepthInterval]
    blocks: list[TextBlock]
