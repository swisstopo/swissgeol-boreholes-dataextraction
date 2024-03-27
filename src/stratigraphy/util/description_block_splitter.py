"""Classes for partitioning material descriptions text into blocks."""

import abc

import fitz
import numpy as np

from stratigraphy.util.dataclasses import Line
from stratigraphy.util.line import TextLine
from stratigraphy.util.textblock import TextBlock


class DescriptionBlockSplitter(metaclass=abc.ABCMeta):
    """Abstract class for splitting material descriptions into blocks based on a certain condition."""

    set_terminated_by_line_flag: bool

    @abc.abstractmethod
    def __init__(self):  # noqa: D107
        pass

    @abc.abstractmethod
    def separator_condition(self, last_line: TextLine, current_line: TextLine, current_block: TextBlock) -> bool:  # noqa: D107
        pass

    def create_blocks(self, description_lines: list[TextLine]) -> list[TextBlock]:
        """Partition the description lines into blocks.

        Args:
            description_lines (list[TextLine]): all the text lines from the material descriptions.

        Returns:
            list[TextBlock]: the list of textblocks
        """
        blocks = []
        current_block_lines = []
        for line in description_lines:
            if len(current_block_lines) > 0:
                last_line = current_block_lines[-1]
                current_block = TextBlock(current_block_lines)
                if self.separator_condition(last_line, line, current_block):
                    current_block.is_terminated_by_line = self.set_terminated_by_line_flag
                    blocks.append(current_block)
                    current_block_lines = []
            current_block_lines.append(line)
        if len(current_block_lines):
            blocks.append(TextBlock(current_block_lines))
        return blocks


class SplitDescriptionBlockByLine(DescriptionBlockSplitter):
    """Creates blocks based on longer lines between the material description text."""

    def __init__(self, threshold: float, material_description_rect: fitz.Rect, geometric_lines: list[Line]):
        """Create a new SplitDescriptionBlockByLine instance.

        Args:
            material_description_rect (fitz.Rect): The bounding box for all material descriptions.
            threshold (float): Percentage of the block width that needs to be covered by a line.
            geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        """
        super().__init__()
        self.threshold = threshold
        self.material_description_rect = material_description_rect
        self.geometric_lines = geometric_lines
        self.set_terminated_by_line_flag = True

    def separator_condition(self, last_line: TextLine, current_line: TextLine, current_block: TextBlock) -> bool:
        """Check if a block is separated by a line.

        Args:
            current_block:
            last_line (TextLine): The previous line.
            current_line (TextLine): The current line.
            current_block (TextBlock): Current block.

        Returns:
            bool: True if the block is separated by a line, False otherwise.
        """
        last_line_y_coordinate = (last_line.rect.y0 + last_line.rect.y1) / 2
        current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2
        for line in self.geometric_lines:
            line_left_x = np.min([line.start.x, line.end.x])
            line_right_x = np.max([line.start.x, line.end.x])
            line_y_coordinate = (line.start.y + line.end.y) / 2

            is_line_long_enough = (
                np.min([self.material_description_rect.x1, line_right_x])
                - np.max([self.material_description_rect.x0, line_left_x])
                > self.threshold * self.material_description_rect.width
            )

            line_ends_block = last_line_y_coordinate < line_y_coordinate < current_line_y_coordinate
            if is_line_long_enough and line_ends_block:
                return True
        return False


class SplitDescriptionBlockByLeftHandSideSeparator(DescriptionBlockSplitter):
    """Creates blocks based on shorter lines at the left-hand side of the material description text."""

    def __init__(self, length_threshold: float, geometric_lines: list[Line]):
        """Create a new SplitDescriptionBlockByLine instance.

        Args:
            length_threshold (int): The minimum length of a line segment on the left side of a block to split it.
            geometric_lines (list[Line]): The geometric lines detected in the pdf page.
        """
        super().__init__()
        self.length_threshold = length_threshold
        self.set_terminated_by_line_flag = False
        self.geometric_lines = geometric_lines

    def separator_condition(self, last_line: TextLine, current_line: TextLine, current_block: TextBlock) -> bool:
        """Check if a block is separated by a line segment on the left side of the block.

        Args:
            last_line (TextLine): The previous line.
            current_line (TextLine): The current line.
            current_block (TextBlock): Current block.

        Returns:
            bool: True if the block is separated by a line segment, False otherwise.
        """
        last_line_y_coordinate = (last_line.rect.y0 + last_line.rect.y1) / 2
        current_line_y_coordinate = (current_line.rect.y0 + current_line.rect.y1) / 2

        for line in self.geometric_lines:
            line_y_coordinate = (line.start.y + line.end.y) / 2

            line_cuts_lefthandside_of_block = line.start.x < current_block.rect.x0 < line.end.x
            is_line_long_enough = (
                np.abs(line.start.x - line.end.x) > self.length_threshold
            )  # for the block splitting, we only care about x-extension

            line_ends_block = last_line_y_coordinate < line_y_coordinate < current_line_y_coordinate

            weak_condition = (
                (line.start.x - 5 < current_block.rect.x0 and line.end.x > current_block.rect.x0)
                and (np.abs(line.start.x - line.end.x) > self.length_threshold - 2)
                and (len(current_block.lines) > 1)
            )  # if block has at least three lines, we weaken the splitting condition.
            # It is three lines because the statement means that block.lines has at least two elements.
            # The third line is current_line
            # The reason for the splitting is, that if we know that a block is long, the probability for
            # a missed splitting is higher. Therefore, we weaken the condition. The weakened condition
            # leads to an improved score as per 21.03.2024.

            if line_ends_block and ((is_line_long_enough and line_cuts_lefthandside_of_block) or weak_condition):
                return True
        return False


class SplitDescriptionBlockByVerticalSpace(DescriptionBlockSplitter):
    """Creates blocks based on vertical spacing between the text lines."""

    def __init__(self, threshold: float):
        """Create a new SplitDescriptionBlockByVerticalSpace instance.

        Args:
            threshold (float): The maximum vertical distance between two lines to be considered part of the same block.
        """
        super().__init__()
        self.threshold = threshold
        self.set_terminated_by_line_flag = False

    def separator_condition(self, last_line: TextLine, current_line: TextLine, current_block: TextBlock) -> bool:
        """Check if a block is separated by sufficient vertical space.

        Args:
            last_line (TextLine): The previous line.
            current_line (TextLine): The current line.
            current_block (TextBlock): Current block.

        Returns:
            bool: True if the block is separated by sufficient vertical space, False otherwise.
        """
        return (
            current_line.rect.y0 > last_line.rect.y1 + 5
            or (  # upper boundary of line is higher up than lower boundary plus 5 points of last line.
                current_line.rect.y0 > last_line.rect.y1 and current_line.rect.y0 > last_line.rect.y0 + self.threshold
            )
        )
