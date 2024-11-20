"""Module for finding AAboveBSidebar instances in a borehole profile."""

import fitz

from stratigraphy.depth import DepthColumnEntryExtractor
from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .a_above_b_sidebar_validator import AAboveBSidebarValidator


class AAboveBSidebarExtractor:
    """Class that finds AAboveBSidebar instances in a borehole profile."""

    @staticmethod
    def find_in_words(
        all_words: list[TextWord], used_entry_rects: list[fitz.Rect], sidebar_params: dict
    ) -> list[AAboveBSidebar]:
        """Construct all possible AAboveBSidebar objects from the given words.

        Args:
            all_words (list[TextLine]): All words in the page.
            used_entry_rects (list[fitz.Rect]): Part of the document to ignore.
            sidebar_params (dict): Parameters for the AAboveBSidebar objects.

        Returns:
            list[AAboveBSidebar]: Found AAboveBSidebar objects.
        """
        entries = [
            entry
            for entry in DepthColumnEntryExtractor.find_in_words(all_words, include_splits=False)
            if entry.rect not in used_entry_rects
        ]

        numeric_columns: list[AAboveBSidebar] = []
        for entry in entries:
            has_match = False
            additional_columns = []
            for column in numeric_columns:
                if column.can_be_appended(entry.rect):
                    has_match = True
                    column.entries.append(entry)
                else:
                    valid_initial_segment = column.valid_initial_segment(entry.rect)
                    if len(valid_initial_segment.entries) > 0:
                        has_match = True
                        valid_initial_segment.entries.append(entry)
                        additional_columns.append(valid_initial_segment)

            numeric_columns.extend(additional_columns)
            if not has_match:
                numeric_columns.append(AAboveBSidebar(entries=[entry]))

            # only keep columns that are not contained in a different column
            numeric_columns = [
                column
                for column in numeric_columns
                if all(not other.strictly_contains(column) for other in numeric_columns)
            ]

        sidebar_validator = AAboveBSidebarValidator(all_words, **sidebar_params)

        numeric_columns = [
            sidebar_validator.reduce_until_valid(column)
            for numeric_column in numeric_columns
            for column in numeric_column.break_on_double_descending()
            # when we have a perfect arithmetic progression, this is usually just a scale
            # that does not match the descriptions
            if not column.significant_arithmetic_progression()
        ]

        return sorted(
            [column for column in numeric_columns if column and sidebar_validator.is_valid(column)],
            key=lambda column: len(column.entries),
        )
