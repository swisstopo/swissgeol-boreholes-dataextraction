"""This module contains logic to validate AAboveBSidebar instances."""

import dataclasses

import rtree

from stratigraphy.lines.line import TextWord

from .a_above_b_sidebar import AAboveBSidebar
from .sidebar import SidebarNoise, noise_count
from .sidebarentry import DepthColumnEntry


@dataclasses.dataclass
class AAboveBSidebarValidator:
    """Validation logic for instances of the AAboveBSidebar class.

    Args:
        noise_count_threshold (float): Noise count threshold deciding how much noise is allowed in a sidebar
                                       to be valid.
        noise_count_offset (int): Offset for the noise count threshold. Affects the noise count criterion.
                                  Effective specifically for sidebars with very few entries.
    """

    noise_count_threshold: float
    noise_count_offset: int

    def is_valid(self, sidebar_noise: SidebarNoise[AAboveBSidebar], corr_coef_threshold: float = 0.99) -> bool:
        """Checks whether the sidebar is valid.

        The sidebar is considered valid if:
        - Its noise_count is less than the noise count threshold
          time the number of entries minus the noise count offset.
        - The entries are strictly increasing.
        - The entries are linearly correlated with their vertical position.

        Args:
            sidebar_noise (SidebarNoise): The SidebarNoise wrapping the sidebar to validate.
            corr_coef_threshold (float): The minimal correlation coefficient for the column to be deemed valid.

        Returns:
            bool: True if the sidebar is valid, False otherwise.
        """
        # When too much other text is in the column, then it is probably not valid.
        # The quadratic behavior of the noise count check makes the check stricter for columns with few entries
        # than columns with more entries. The more entries we have, the less likely it is that we found them by chance.
        # TODO: Once evaluation data is of good enough qualities, we should optimize for the parameter below.

        sidebar = sidebar_noise.sidebar
        noise = sidebar_noise.noise_count

        if noise > self.noise_count_threshold * (len(sidebar.entries) - self.noise_count_offset) ** 2:
            return False
        # Check if the entries are strictly increasing.
        if not sidebar.is_strictly_increasing():
            return False
        if sidebar.close_to_arithmetic_progression():
            return False

        corr_coef = sidebar.pearson_correlation_coef()

        return corr_coef and corr_coef > corr_coef_threshold

    def reduce_until_valid(
        self, sidebar_noise: SidebarNoise[AAboveBSidebar], all_words: list[TextWord], word_rtree: rtree.index.Index
    ) -> SidebarNoise | None:
        """Removes entries from the depth column until it fulfills the is_valid condition.

        is_valid checks whether there is too much noise (i.e. other text) in the column and whether the entries are
        linearly correlated with their vertical position.

        Args:
            sidebar_noise (SidebarNoise): The SidebarNoise wrapping the AAboveBSidebar to validate.
            all_words (list[TextWord]): A list of all words contained on a page.
            word_rtree (rtree.index.Index): Pre-built R-tree for spatial queries.

        Returns:
            sidebar_noise | None : The current SidebarNoise with entries removed from Sidebar until it is valid
            and the recalculated noise_count or None.
        """
        while sidebar_noise.sidebar.entries:
            if self.is_valid(sidebar_noise):
                return sidebar_noise

            corrected_sidebar_noise = self.correct_OCR_mistakes(sidebar_noise, all_words, word_rtree)
            if corrected_sidebar_noise:
                return corrected_sidebar_noise

            new_sidebar = sidebar_noise.sidebar.remove_entry_by_correlation_gradient()
            if not new_sidebar:
                return None

            new_noise_count = noise_count(new_sidebar, all_words, word_rtree)
            sidebar_noise = SidebarNoise(sidebar=new_sidebar, noise_count=new_noise_count)

        return None

    def correct_OCR_mistakes(
        self, sidebar_noise: SidebarNoise, all_words: list[TextWord], word_rtree: rtree.index.Index
    ) -> SidebarNoise | None:
        """Corrects OCR mistakes in the Sidebar entries.

        Loops through all values and corrects common OCR mistakes for the given entry. Then, the column with the
        highest pearson correlation coefficient is selected and checked for validity.

        This is useful if one or more entries have an OCR mistake, and the column is not valid because of it.

        Currently, there is no limit on the number of corrections per depth column. Indeed, there are examples of depth
        columns with multiple OCR errors on different depth values. On the other hand, allowing an unlimited number of
        corrections increases the risk, that a random column of different values is incorrectly accepted as a depth
        column after making the corrections, especially if the column has a low number of entries. A more robust
        solution might be to allow corrections on less than 50% of all entries, or something similar. However, we
        currently don't have enough examples to properly tune this parameter.

        Note: Common mistakes should be extended as needed.

        Args:
            sidebar_noise (SidebarNoise): The SidebarNoise wrapping the sidebar to validate.
            all_words (list[TextWord]): All words on the page for recalculating noise count.
            word_rtree (index.Index): R-tree for efficient spatial queries.

        Returns:
            SidebarNoise | None: The corrected SidebarNoise, or None if no correction was possible.
        """
        sidebar = sidebar_noise.sidebar
        new_columns = [AAboveBSidebar(entries=[])]

        for entry in sidebar.entries:
            new_columns = [
                AAboveBSidebar([*column.entries, DepthColumnEntry(entry.rect, new_value)])
                for column in new_columns
                for new_value in _value_alternatives(entry.value)
            ]
            # Immediately require strictly increasing values, to avoid exponential complexity when many implausible
            # alternative values are suggested
            new_columns = [column for column in new_columns if column.is_strictly_increasing()]

        if new_columns:
            best_column = max(new_columns, key=lambda column: column.pearson_correlation_coef())
            new_noise_count = noise_count(best_column, all_words, word_rtree)

            # We require a higher correlation coefficient when corrections are made
            if self.is_valid(
                SidebarNoise(sidebar=best_column, noise_count=new_noise_count), corr_coef_threshold=0.999
            ):
                return SidebarNoise(sidebar=best_column, noise_count=new_noise_count)

        return None


def _value_alternatives(value: float) -> set[float]:
    """Corrects frequent OCR errors in depth column entries.

    Args:
        value (float): The depth values to find plausible alternatives for

    Returns:
        set(float): all plausible values (including the original one)
    """
    alternatives = {value}
    # In older documents, OCR sometimes mistakes 1 for 4
    alternatives.add(float(str(value).replace("4", "1")))

    return alternatives
